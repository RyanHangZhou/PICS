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


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
    return gen_image


def crop_back2(pred, tar_image, extra_sizes, tar_box_yyxx_crop, tar_box_yyxx_crop2):
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
    cond_key = 'view'
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

    guidance_scale = 9.0 #5.0 #1.0 #5.0 # 9.0

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

    pred = crop_back2(pred, back_image, sizes, sizes0, sizes1) 
    return pred

def process_pairs_multiple(mask, tar_image, patch_dir, counter=0, max_ratio=0.8):

    view = cv2.cvtColor(cv2.imread(patch_dir), cv2.COLOR_BGR2RGB)
    view = expand_image(view)
    view = pad_to_square(view, pad_value = 255, random = False) # pad to square
    view = cv2.resize(view.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1

    # ========= Training Target ===========
    box_yyxx = get_bbox_from_mask(mask) # mask: [1024, 768], [267, 903, 37, 759]
    # print('box_yyxx: ', box_yyxx) 
    box_yyxx = expand_bbox(mask, box_yyxx, ratio=[1.1,1.2]) #1.1  1.3,    [234, 935, 0, 768]
    # print('after box_yyxx: ', box_yyxx)
    
    box_yyxx_crop = [0, tar_image.shape[0], 0, tar_image.shape[1]]
    y1,y2,x1,x2 = box_yyxx_crop # [0 1024 0 768]

    cropped_target_image = tar_image[y1:y2,x1:x2,:] # [1024, 768, 3]
    cropped_tar_mask = mask[y1:y2,x1:x2] # # [1024, 768]
    box_yyxx = box_in_box(box_yyxx, box_yyxx_crop)
    y1,y2,x1,x2 = box_yyxx # [234 935 0 768]

    collage = cropped_target_image.copy() # [1024, 768, 3]
    source_collage = collage
    collage[y1:y2,x1:x2,:] = 0

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    H1, W1 = collage.shape[0], collage.shape[1]

    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
    H2, W2 = collage.shape[0], collage.shape[1]

    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
    collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
    collage_mask[collage_mask == 2] = -1

    collage_mask_ = collage_mask
    collage_mask_[collage_mask_ == -1] = 0
    collage_mask_ = collage_mask_[:, :, 0]

    view = view/255.
    
    # Prepairing dataloader items
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    collage_mask_ = np.expand_dims(collage_mask_, axis=-1)
    
    item = {}
    item['hint'+str(counter)] = {}
    item['jpg'] = {}
    item['extra_sizes'] = {}

    item.update({'view'+str(counter): view.copy()}) # patch 0, 1 (checked) [0, 1], 224x224x3

    item.update({'jpg': cropped_target_image.copy()})
    item.update({'hint'+str(counter): collage.copy()})
    item.update({'mask'+str(counter): collage_mask_.copy()}) # ref_mask_compose
    item.update({'collage': source_collage.copy()})
    item.update({'extra_sizes': np.array([H1, W1, H2, W2])})
    item.update({'hint_sizes'+str(counter): np.array([y1, x1, y2, x2])})

    return item


def process_composition(item, obj_thr):

    collage = item['collage']
    collage_mask = collage.copy() * 0.0
    collage_edges = collage.copy() * 0.0

    for i in reversed(range(obj_thr)):
        y1, x1, y2, x2 = item['hint_sizes'+str(i)]
        collage[y1:y2, x1:x2, :] = 0
        collage_mask[y1:y2,x1:x2,:] = 1.0

    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
    # cv2.imwrite('collage.png', collage)
    collage = collage / 127.5 - 1.0 

    collage_mask = pad_to_square(collage_mask, pad_value = 0, random = False).astype(np.uint8)
    collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)

    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
    # print('collage max min: ', np.max(collage), np.min(collage)) # [2], [-1]
    # import pdb; pdb.set_trace()
    # print('collage shape: ', np.shape(collage)) # [512, 512, 4]
    # wwwwww
    item.update({'hint': collage.copy()})
    return item


def crop_to_match_reference(gen_image, back_image):
    """
    根据参考图像的比例裁剪生成图像（移除黑色区域）。
    :param gen_image: 生成图像，形状为 (H, W, 3)。
    :param back_image: 对比图像，形状为 (H_ref, W_ref, 3)。
    :return: 根据对比图像裁剪后的生成图像。
    """
    # 获取生成图像和对比图像的尺寸
    gen_h, gen_w, _ = gen_image.shape
    ref_h, ref_w, _ = back_image.shape

    # 计算参考图的宽高比
    ref_aspect = ref_w / ref_h
    gen_aspect = gen_w / gen_h

    if gen_aspect > ref_aspect:
        # 生成图宽度过大，需要裁剪左右
        new_width = int(gen_h * ref_aspect)
        start_x = (gen_w - new_width) // 2
        cropped_image = gen_image[:, start_x:start_x + new_width]
    elif gen_aspect < ref_aspect:
        # 生成图高度过大，需要裁剪上下
        new_height = int(gen_w / ref_aspect)
        start_y = (gen_h - new_height) // 2
        cropped_image = gen_image[start_y:start_y + new_height, :]
    else:
        # 比例相等，无需裁剪
        cropped_image = gen_image

    return cropped_image


if __name__ == '__main__': 

    sample_num = 6701
    compose_num = 2
    obj_thr = 2
    save_path = '/home/hang18/links/projects/rrg-vislearn/hang18/COIN/results/lvis'
    input_path = '/home/hang18/links/projects/rrg-vislearn/hang18/COIN/data/test/LVIS'
    os.makedirs(save_path, exist_ok=True)
    img_list = os.listdir(input_path)
    for k in range(1):
        # for i in range(0, sample_num):
        # for t in img_list[4417:sample_num]:
        for t in img_list[2117:sample_num]:
        # for t in img_list:
            try: 
            # if True:
                i = t
                print(os.path.join(input_path, i+'/'+'image.jpg'))
                back_image = cv2.imread(os.path.join(input_path, i+'/'+'image.jpg')).astype(np.uint8)
                back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

                counter = 0
                item_with_collage = {}
                for j in range(compose_num):
                    patch_dir = os.path.join(input_path, i+"/object_"+str(j)+".png")
                    bg_mask_path_i = os.path.join(input_path, i+"/object_"+str(j)+"_mask.png")
                    tar_mask = cv2.imread(bg_mask_path_i)[:,:,0] > 128
                    tar_mask = tar_mask.astype(np.uint8)

                    item = process_pairs_multiple(tar_mask, back_image.copy(), patch_dir, counter)
                    item_with_collage.update(item)
                    counter += 1

                item_with_collage = process_composition(item_with_collage, obj_thr)
                gen_image = inference_single_image_multi(item_with_collage, back_image)
                
                comp_image_dir = os.path.join(save_path, 'composed')
                if not os.path.exists(comp_image_dir):
                    os.mkdir(comp_image_dir)
                comp_save_path = os.path.join(comp_image_dir, 'composed'+str(i)+'__'+str(k)+'.png')
                cv2.imwrite(comp_save_path, gen_image[:,:,::-1])
                # wwww

            except: 
                print('fail')


