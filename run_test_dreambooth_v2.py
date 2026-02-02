import cv2
import os
import einops
import numpy as np
import torch
import random
import sys
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
from pycocotools.coco import COCO
from lvis import LVIS


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def aug_patch(patch):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE),
        ])

    transformed = transform(image=patch)
    transformed_patch = transformed["image"]
    return transformed_patch

def sample_timestep(max_step = 1000):
    if np.random.rand() < 0.3:
        step = np.random.randint(0,max_step)
        return np.array([step])
    step_start = 0
    step_end = max_step
    step = np.random.randint(step_start, step_end)
    return np.array([step])

def _construct_collage(image, object_0, object_1, mask_0, mask_1):
    background = image.copy()
    image = pad_to_square(image, pad_value = 0, random = False).astype(np.uint8)
    image = cv2.resize(image.astype(np.uint8), (512,512)).astype(np.float32)
    image = image / 127.5 - 1.0
    item = {}
    item.update({'jpg': image.copy()}) # source image (checked) [-1, 1], 512x512x3

    ratio = np.random.randint(11, 15) / 10 
    object_0 = expand_image(object_0, ratio=ratio)
    object_0 = aug_patch(object_0)
    object_0 = pad_to_square(object_0, pad_value = 255, random = False) # pad to square
    object_0 = cv2.resize(object_0.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1
    object_0 = object_0  / 255
    item.update({'ref0': object_0.copy()}) # patch 0 (checked) [0, 1], 224x224x3

    ratio = np.random.randint(11, 15) / 10 
    object_1 = expand_image(object_1, ratio=ratio)
    object_1 = aug_patch(object_1)
    object_1 = pad_to_square(object_1, pad_value = 255, random = False) # pad to square
    object_1 = cv2.resize(object_1.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1
    object_1 = object_1  / 255 
    item.update({'ref1': object_1.copy()}) # patch 1 (checked) [0, 1], 224x224x3

    background_mask0 = background.copy() * 0.0
    background_mask1 = background.copy() * 0.0
    background_mask = background.copy() * 0.0

    box_yyxx = get_bbox_from_mask(mask_0)
    box_yyxx = expand_bbox(mask_0, box_yyxx, ratio=[1.1, 1.2]) #1.1  1.3
    y1, y2, x1, x2 = box_yyxx
    background[y1:y2, x1:x2,:] = 0
    background_mask0[y1:y2, x1:x2, :] = 1.0
    background_mask[y1:y2, x1:x2, :] = 1.0
    item.update({'hint_sizes'+str(0): np.array([y1, x1, y2, x2])})

    box_yyxx = get_bbox_from_mask(mask_1)
    box_yyxx = expand_bbox(mask_0, box_yyxx, ratio=[1.1, 1.2]) #1.1  1.3
    y1, y2, x1, x2 = box_yyxx
    background[y1:y2, x1:x2,:] = 0
    background_mask1[y1:y2, x1:x2, :] = 1.0
    background_mask[y1:y2, x1:x2, :] = 1.0
    item.update({'hint_sizes'+str(1): np.array([y1, x1, y2, x2])})

    H1, W1 = background.shape[0], background.shape[1]

    background = pad_to_square(background, pad_value = 0, random = False).astype(np.uint8)
    H2, W2 = background.shape[0], background.shape[1]
    background = cv2.resize(background.astype(np.uint8), (512,512)).astype(np.float32)
    background_mask0 = pad_to_square(background_mask0, pad_value = 2, random = False).astype(np.uint8)
    background_mask1 = pad_to_square(background_mask1, pad_value = 2, random = False).astype(np.uint8)
    background_mask = pad_to_square(background_mask, pad_value = 2, random = False).astype(np.uint8)
    background_mask0  = cv2.resize(background_mask0.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
    background_mask1  = cv2.resize(background_mask1.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
    background_mask  = cv2.resize(background_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
    
    background_mask0[background_mask0 == 2] = -1
    background_mask1[background_mask1 == 2] = -1
    background_mask[background_mask == 2] = -1

    background_mask0_ = background_mask0
    background_mask0_[background_mask0_ == -1] = 0
    background_mask0_ = background_mask0_[:, :, 0]

    background_mask1_ = background_mask1
    background_mask1_[background_mask1_ == -1] = 0
    background_mask1_ = background_mask1_[:, :, 0]

    background = background / 127.5 - 1.0 
    background = np.concatenate([background, background_mask[:,:,:1]] , -1)
    item.update({'hint': background.copy()}) # condition image (temporal) [-1, 1], 512x512x4

    background_mask0_ = np.expand_dims(background_mask0_, axis=-1)
    background_mask1_ = np.expand_dims(background_mask1_, axis=-1)

    item.update({'mask0': background_mask0_.copy()}) # mask (checked) [0, 1], 512x512
    item.update({'mask1': background_mask1_.copy()}) # mask (checked) [0, 1], 512x512

    sampled_time_steps = sample_timestep()
    item['time_steps'] = sampled_time_steps
    item['object_num'] = 2
    item.update({'extra_sizes': np.array([H1, W1, H2, W2])})

    return item


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop, tar_box_yyxx_crop2):
    H1, W1, H2, W2 = extra_sizes
    y1,x1,y2,x2 = tar_box_yyxx_crop
    y1_,x1_, y2_,x2_ = tar_box_yyxx_crop2
    m = 0 # maigin_pixel

    if H1 < W1:
        pad1 = int((W1 - H1) / 2)
        pad2 = W1 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]
    elif H1 > W1:
        pad1 = int((H1 - W1) / 2)
        pad2 = H1 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] = pred[y1+m :y2-m, x1+m:x2-m, :]
    gen_image[y1_+m :y2_-m, x1_+m:x2_-m, :] = pred[y1_+m :y2_-m, x1_+m:x2_-m, :]
    return gen_image


def aug_patch(patch):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE),
        ])

    transformed = transform(image=patch)
    transformed_patch = transformed["image"]
    return transformed_patch

def get_input(batch, k):
    # print(k)
    # import pdb; pdb.set_trace()
    x = batch[k]
    if len(x.shape) == 3:
        x = x[None, ...]

    x = torch.tensor(x)
    x = einops.rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()
    return x

def get_unconditional_conditioning(N, obj_thr):
    # uncond = self.get_learned_conditioning([ torch.zeros((1, 3, 224, 224)) ] * N)
    x = [ torch.zeros((1, 3, 224, 224)) ] * N
    uc = []
    for i in range(obj_thr):
        uc_i = model.get_learned_conditioning(x)
        uc.append(uc_i)
    uc = torch.stack(uc)
    uc = uc.permute(1, 2, 3, 0)
    return {"pch_code": uc}

def inference_single_image_multi(item, back_image):
    obj_thr = 2
    cond_key = 'ref'
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    xc = []
    xc_mask = []
    for obj_i in range(obj_thr):
        cond_key_i = cond_key+str(obj_i)
        mask_cond_key_i = "mask" + str(obj_i)

        xc_i = get_input(item, cond_key_i)

        xc.append(xc_i)

        mask_xc_i = get_input(item, mask_cond_key_i)
        xc_mask.append(mask_xc_i) 

    c = []
    for xc_i in xc:
        c_i = model.get_learned_conditioning(xc_i.cuda())
        c.append(c_i)

    c = torch.stack(c)
    c = c.permute(1, 2, 3, 0)
    c = {
        "pch_code": c
    }

    c_mask = torch.stack(xc_mask)
    c_mask = c_mask.permute(1, 2, 3, 4, 0)

    guidance_scale = 5.0 #1.0 #5.0 # 9.0

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    # ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    # ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    # print('hint max min: ', np.max(hint), np.min(hint)) # [2], [-1]
    # import pdb; pdb.set_trace()
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda()  # xxxx
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone() # [512, 512, 4] xxx
    # import pdb; pdb.set_trace()

    guess_mode = False
    H,W = 512,512

    # cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    cond = {"c_concat": [control], "c_crossattn": [c], "c_mask": [c_mask]}

    # c = []
    # for xc_i in xc:
    #     c_i = model.get_learned_conditioning(xc_i.to(self.device))
    #     # print('c_i shape: ', np.shape(c_i))
    #     c.append(c_i)

    # un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)], "c_mask": [c_mask]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [get_unconditional_conditioning(num_samples, obj_thr)], "c_mask": [c_mask]}

    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,:]
    result = np.clip(result,0,255)

    # import pdb; pdb.set_trace()

    pred = x_samples[0]
    # print(np.shape(np.clip(pred,0,255)))
    # import pdb; pdb.set_trace()
    pred = np.clip(pred,0,255)
    sizes = item['extra_sizes']
    sizes0 = item['hint_sizes0']
    sizes1 = item['hint_sizes1']
    # tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    # gen_image = crop_back(pred, back_image, sizes, tar_box_yyxx_crop)
    # import pdb; pdb.set_trace()
    if back_image.shape[0] <= back_image.shape[1]:
        pred = cv2.resize(pred, (back_image.shape[1], back_image.shape[1]))
    else:
        pred = cv2.resize(pred, (back_image.shape[0], back_image.shape[0]))

    pred = crop_back(pred, back_image, sizes, sizes0, sizes1) 
    return pred


if __name__ == '__main__': 

    sample_num = 79
    compose_num = 2
    obj_thr = 2
    # save_path = '/home/hang18/links/projects/rrg-vislearn/hang18/COIN/results/dreambooth'
    # save_path = '/home/hang18/links/projects/rrg-vislearn/hang18/COIN/results/dreambooth_lvis'
    # save_path = '/home/hang18/links/projects/rrg-vislearn/hang18/COIN/results/dreambooth_lvis_attn2_v2'
    save_path = '/home/hang18/links/projects/rrg-vislearn/hang18/COIN/results/dreambooth_lvis_attn2_v2_ep13'
    input_path = '/home/hang18/links/projects/rrg-vislearn/hang18/COIN/data/test/dreambooth'
    os.makedirs(save_path, exist_ok=True)
    img_list = os.listdir(input_path)
    for k in range(1):
        for t in img_list:

            try: 
                sample_path = os.path.join(input_path, t)
                image = cv2.cvtColor(cv2.imread(os.path.join(sample_path, "image.jpg")), cv2.COLOR_BGR2RGB)
                object_0 = cv2.cvtColor(cv2.imread(os.path.join(sample_path, "object_0.png")), cv2.COLOR_BGR2RGB)
                object_1 = cv2.cvtColor(cv2.imread(os.path.join(sample_path, "object_1.png")), cv2.COLOR_BGR2RGB)
                mask_0 = cv2.imread(os.path.join(sample_path, "object_0_mask.png"), cv2.IMREAD_GRAYSCALE)
                mask_1 = cv2.imread(os.path.join(sample_path, "object_1_mask.png"), cv2.IMREAD_GRAYSCALE)
                collage = _construct_collage(image, object_0, object_1, mask_0, mask_1)

                # item_with_collage = process_composition(collage, obj_thr)
                gen_image = inference_single_image_multi(collage, image)
                
                comp_image_dir = os.path.join(save_path, 'composed')
                if not os.path.exists(comp_image_dir):
                    os.mkdir(comp_image_dir)
                comp_save_path = os.path.join(comp_image_dir, 'composed'+str(t)+'__'+str(k)+'.png')
                cv2.imwrite(comp_save_path, gen_image[:,:,::-1])

            except: 
                print('fail')


