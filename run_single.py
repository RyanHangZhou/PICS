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
import shutil


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


# config = OmegaConf.load('./configs/inference.yaml')
# model_ckpt =  config.pretrained_model
# model_config = config.config_file

# model = create_model(model_config ).cpu()
# model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
# model = model.cuda()
# ddim_sampler = DDIMSampler(model)



def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug 
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image) 

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255) # [224, 224, 3]
    # import pdb; pdb.set_trace()

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    # cv2.imwrite('bin/4.png', cropped_target_image)
    # print(y1, y2, x1, x2) # 0, 500, 0, 461
    # print(np.shape(cropped_target_image))
    # www
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx # this one is the location of the target composition!

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1) # combination of sobel (3-channel) and mask (1-channel)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 
    return item


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


def get_input(batch, k):
    # print(k)
    # import pdb; pdb.set_trace()
    x = batch[k]
    if len(x.shape) == 3:
        x = x[None, ...]
    # print('ww', np.shape(x)) # [1, 224, 224, 3]
    # import pdb; pdb.set_trace()

    x = torch.tensor(x)
    x = einops.rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()
    return x


# def get_learned_conditioning(c):
#     #c 1,3,224,224 
#     if self.cond_stage_forward is None:
#         if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
#             #1,1,1024
#             c = self.cond_stage_model.encode(c)
#             if isinstance(c, DiagonalGaussianDistribution):
#                 c = c.mode()
#         else:
#             c = self.cond_stage_model(c)
#     else:
#         assert hasattr(self.cond_stage_model, self.cond_stage_forward)
#         c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
#     return c

def get_unconditional_conditioning(N, obj_thr):
    # uncond = self.get_learned_conditioning([ torch.zeros((1, 3, 224, 224)) ] * N)
    x = [ torch.zeros((1, 3, 224, 224)) ] * N
    uc = []
    for i in range(obj_thr):
        uc_i = model.get_learned_conditioning(x)
        uc.append(uc_i)
    uc = torch.stack(uc)
    uc = uc.permute(1, 2, 3, 0)
    return uc

def inference_single_image_multi(item, back_image):
    obj_thr = 2
    # cond_key = 'ref'
    cond_key = 'view'
    # ref0 = item['ref0'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    # xc = []
    xc0 = []
    xc1 = []
    xc2 = []
    xc3 = []
    xc4 = []
    xc5 = []
    xc_mask = []
    for obj_i in range(obj_thr):
        # cond_key_i = cond_key+str(obj_i)
        cond_key0_i = cond_key+'0'+str(obj_i)
        cond_key1_i = cond_key+'1'+str(obj_i)
        cond_key2_i = cond_key+'2'+str(obj_i)
        cond_key3_i = cond_key+'3'+str(obj_i)
        cond_key4_i = cond_key+'4'+str(obj_i)
        cond_key5_i = cond_key+'5'+str(obj_i)
        mask_cond_key_i = "mask" + str(obj_i)

        # xc_i = get_input(item, cond_key_i)
        xc0_i = get_input(item, cond_key0_i)
        xc1_i = get_input(item, cond_key1_i)
        xc2_i = get_input(item, cond_key2_i)
        xc3_i = get_input(item, cond_key3_i)
        xc4_i = get_input(item, cond_key4_i)
        xc5_i = get_input(item, cond_key5_i)

        # print(np.shape(xc_i)) # [1, 224, 224, 3]
        # import pdb; pdb.set_trace()
        # xc.append(xc_i)
        xc0.append(xc0_i.cpu())
        xc1.append(xc1_i.cpu())
        xc2.append(xc2_i.cpu())
        xc3.append(xc3_i.cpu())
        xc4.append(xc4_i.cpu())
        xc5.append(xc5_i.cpu())

        mask_xc_i = get_input(item, mask_cond_key_i)
        # print(np.shape(mask_xc_i)) # [1, 512, 1, 512]
        # import pdb; pdb.set_trace()
        xc_mask.append(mask_xc_i) 

    # c = []
    # for xc_i in xc:
    #     # print(np.shape(xc_i)) # torch.Size([224, 1, 224, 3])
    #     # import pdb; pdb.set_trace()
    #     c_i = model.get_learned_conditioning(xc_i.cuda())
    #     # print('c_i shape: ', np.shape(c_i)) # [[1, 224, 224, 3]]
    #     c.append(c_i)

    c0 = []
    for xc_i in xc0:
        c_i = model.get_learned_conditioning(xc_i.cuda())
        c0.append(c_i)
    c1 = []
    for xc_i in xc1:
        c_i = model.get_learned_conditioning(xc_i.cuda())
        c1.append(c_i)
    c2 = []
    for xc_i in xc2:
        c_i = model.get_learned_conditioning(xc_i.cuda())
        c2.append(c_i)
    c3 = []
    for xc_i in xc3:
        c_i = model.get_learned_conditioning(xc_i.cuda())
        c3.append(c_i)
    c4 = []
    for xc_i in xc4:
        c_i = model.get_learned_conditioning(xc_i.cuda())
        c4.append(c_i)
    c5 = []
    for xc_i in xc5:
        c_i = model.get_learned_conditioning(xc_i.cuda())
        c5.append(c_i)

    # c = torch.stack(c)
    c0 = torch.stack(c0)
    c1 = torch.stack(c1)
    c2 = torch.stack(c2)
    c3 = torch.stack(c3)
    c4 = torch.stack(c4)
    c5 = torch.stack(c5)
    # c = c.permute(1, 2, 3, 0)
    c0 = c0.permute(1, 2, 3, 0)
    c1 = c1.permute(1, 2, 3, 0)
    c2 = c2.permute(1, 2, 3, 0)
    c3 = c3.permute(1, 2, 3, 0)
    c4 = c4.permute(1, 2, 3, 0)
    c5 = c5.permute(1, 2, 3, 0)
    c = {
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c5": c5
    }

    c_mask = torch.stack(xc_mask)
    c_mask = c_mask.permute(1, 2, 3, 4, 0)

    guidance_scale = 5.0

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


    # clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    # clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    # clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone() # [224, 224, 3]

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

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    # import pdb; pdb.set_trace()

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, back_image, sizes, tar_box_yyxx_crop) 
    return gen_image


def download_image(img_id):
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']
    img_path = os.path.join(img_dir, img_info['file_name'])
    return img_path

def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
        ])

    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask

def process_pairs_multiple(ref_image, ref_mask, tar_image, tar_mask, patch_dir, counter=0, max_ratio=0.8):
    # assert mask_score(ref_mask) > 0.90
    # assert self.check_mask_area(ref_mask) == True
    # assert self.check_mask_area(tar_mask)  == True

    # ========= Reference ===========
    '''
    # similate the case that the mask for reference object is coarse. Seems useless :(

    if np.random.uniform(0, 1) < 0.7: 
        ref_mask_clean = ref_mask.copy()
        ref_mask_clean = np.stack([ref_mask_clean,ref_mask_clean,ref_mask_clean],-1)
        ref_mask = perturb_mask(ref_mask, 0.6, 0.9)
        
        # select a fake bg to avoid the background leakage
        fake_target = tar_image.copy()
        h,w = ref_image.shape[0], ref_image.shape[1]
        fake_targe = cv2.resize(fake_target, (w,h))
        fake_back = np.fliplr(np.flipud(fake_target))
        fake_back = self.aug_data_back(fake_back)
        ref_image = ref_mask_clean * ref_image + (1-ref_mask_clean) * fake_back
    '''

    # Get the outline Box of the reference image
    ref_box_yyxx = get_bbox_from_mask(ref_mask) # y1y2x1x2, obtain location from ref patch
    # assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
    
    # Filtering background for the reference image
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) # obtain patch (outside white)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2] # obtain a tight mask

    # print(patch_dir)
    # import pdb; pdb.set_trace()
    multiview0 = cv2.cvtColor(cv2.imread(os.path.join(patch_dir, '0.jpg')), cv2.COLOR_BGR2RGB)
    multiview1 = cv2.cvtColor(cv2.imread(os.path.join(patch_dir, '1.jpg')), cv2.COLOR_BGR2RGB)
    multiview2 = cv2.cvtColor(cv2.imread(os.path.join(patch_dir, '2.jpg')), cv2.COLOR_BGR2RGB)
    multiview3 = cv2.cvtColor(cv2.imread(os.path.join(patch_dir, '3.jpg')), cv2.COLOR_BGR2RGB)
    multiview4 = cv2.cvtColor(cv2.imread(os.path.join(patch_dir, '4.jpg')), cv2.COLOR_BGR2RGB)
    multiview5 = cv2.cvtColor(cv2.imread(os.path.join(patch_dir, '5.jpg')), cv2.COLOR_BGR2RGB)

    ratio = np.random.randint(11, 15) / 10 
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1) # dilated patch mask

    multiview0 = expand_image(multiview0)
    multiview1 = expand_image(multiview1)
    multiview2 = expand_image(multiview2)
    multiview3 = expand_image(multiview3)
    multiview4 = expand_image(multiview4)
    multiview5 = expand_image(multiview5)

    # Padding reference image to square and resize to 224
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) # pad to square
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1

    multiview0 = pad_to_square(multiview0, pad_value = 255, random = False) # pad to square
    multiview0 = cv2.resize(multiview0.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1
    multiview1 = pad_to_square(multiview1, pad_value = 255, random = False) # pad to square
    multiview1 = cv2.resize(multiview1.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1
    multiview2 = pad_to_square(multiview2, pad_value = 255, random = False) # pad to square
    multiview2 = cv2.resize(multiview2.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1
    multiview3 = pad_to_square(multiview3, pad_value = 255, random = False) # pad to square
    multiview3 = cv2.resize(multiview3.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1
    multiview4 = pad_to_square(multiview4, pad_value = 255, random = False) # pad to square
    multiview4 = cv2.resize(multiview4.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1
    multiview5 = pad_to_square(multiview5, pad_value = 255, random = False) # pad to square
    multiview5 = cv2.resize(multiview5.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0] # check 2

    # Augmenting reference image
    #masked_ref_image_aug = self.aug_data(masked_ref_image) 
    
    # Getting for high-freqency map
    # masked_ref_image_compose, ref_mask_compose = aug_data_mask(masked_ref_image, ref_mask)
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask
    masked_ref_image_aug = masked_ref_image_compose.copy()

    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
    # cv2.imwrite('ref_image_collage.png', ref_image_collage*255)
    # print(np.max(ref_image_collage), np.min(ref_image_collage)) # [225, 0], [184, 0]
    # print(np.shape(ref_image_collage)) # [224, 224, 3]
    # wwww
    

    # ========= Training Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
    # assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True
    
    # Cropping around the target object 
    # tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
    # tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    # y1,y2,x1,x2 = tar_box_yyxx_crop
    tar_box_yyxx_crop = [0, tar_image.shape[0], 0, tar_image.shape[1]]
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    cropped_tar_mask = tar_mask[y1:y2,x1:x2]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # Prepairing collage image
    ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
    # ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    # ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy()
    source_collage = collage
    collage[y1:y2,x1:x2,:] = 0 #ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # if np.random.uniform(0, 1) < 0.7: 
    #     cropped_tar_mask = perturb_mask(cropped_tar_mask)
    #     collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

    H1, W1 = collage.shape[0], collage.shape[1]

    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
    H2, W2 = collage.shape[0], collage.shape[1]

    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
    collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
    collage_mask[collage_mask == 2] = -1

    ref_mask_compose = ref_mask_compose/255.
    collage_mask_ = collage_mask
    collage_mask_[collage_mask_ == -1] = 0
    collage_mask_ = collage_mask_[:, :, 0]

    multiview0 = multiview0/255.
    multiview1 = multiview1/255.
    multiview2 = multiview2/255.
    multiview3 = multiview3/255.
    multiview4 = multiview4/255.
    multiview5 = multiview5/255.
    
    # Prepairing dataloader items
    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    # collage_mask_ = collage_mask
    # collage_mask_[collage_mask_ == -1] = 0
    # collage_mask_ = collage_mask_[:, :, 0]
    collage_mask_ = np.expand_dims(collage_mask_, axis=-1)


    # ref_key = f"ref{counter}"
    # hint_key = f"hint{counter}"
    
    item = {}
    item['ref'+str(counter)] = {}
    item['hint'+str(counter)] = {}
    item['jpg'] = {}
    item['extra_sizes'] = {}
    item['tar_box_yyxx_crop'] = {}
    # print('counter:', counter)

    item.update({'ref'+str(counter): masked_ref_image_aug.copy()})
    item.update({'view0'+str(counter): multiview0.copy()}) # patch 0, 1 (checked) [0, 1], 224x224x3
    item.update({'view1'+str(counter): multiview1.copy()}) # patch 0, 1 (checked) [0, 1], 224x224x3
    item.update({'view2'+str(counter): multiview2.copy()}) # patch 0, 1 (checked) [0, 1], 224x224x3
    item.update({'view3'+str(counter): multiview3.copy()}) # patch 0, 1 (checked) [0, 1], 224x224x3
    item.update({'view4'+str(counter): multiview4.copy()}) # patch 0, 1 (checked) [0, 1], 224x224x3
    item.update({'view5'+str(counter): multiview5.copy()}) # patch 0, 1 (checked) [0, 1], 224x224x3
    item.update({'jpg': cropped_target_image.copy()})
    item.update({'hint'+str(counter): collage.copy()})
    item.update({'mask'+str(counter): collage_mask_.copy()}) # ref_mask_compose
    item.update({'collage': source_collage.copy()})
    item.update({'ref_patch'+str(counter): ref_image_collage.copy()})
    item.update({'extra_sizes': np.array([H1, W1, H2, W2])})
    item.update({'hint_sizes'+str(counter): np.array([y1, x1, y2, x2])})
    item.update({'tar_box_yyxx_crop': np.array(tar_box_yyxx_crop)})

    # item = dict(
    #         ref_key=masked_ref_image_aug.copy(), 
    #         jpg=cropped_target_image.copy(), 
    #         hint_key=collage.copy(), 
    #         extra_sizes=np.array([H1, W1, H2, W2]), 
    #         tar_box_yyxx_crop=np.array(tar_box_yyxx_crop) 
    #         )
    # print(item)
    # wwww
    return item


def process_composition(item, obj_thr):

    collage = item['collage']
    collage_mask = collage.copy() * 0.0
    collage_edges = collage.copy() * 0.0

    # for i in reversed(range(obj_thr)):
    #     ref_patch_i = item['ref_patch'+str(i)]
    #     y1, x1, y2, x2 = item['hint_sizes'+str(i)]
    #     collage[y1:y2, x1:x2, :] = ref_patch_i
    #     collage_mask[y1:y2,x1:x2,:] = 1.0
    for i in reversed(range(obj_thr)):
        ref_patch_i = item['ref_patch'+str(i)]
        y1, x1, y2, x2 = item['hint_sizes'+str(i)]
        collage[y1:y2, x1:x2, :] = 0#ref_patch_i
        collage_mask[y1:y2,x1:x2,:] = 1.0

    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
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


if __name__ == '__main__': 
    # '''
    # ==== Example for inferring a single image ===
    # reference_image_path = './examples/TestDreamBooth/FG/01.png'
    # bg_image_path = './examples/TestDreamBooth/BG/000000309203_GT.png'
    # bg_mask_path = './examples/TestDreamBooth/BG/000000309203_mask.png'
    # save_path = './examples/TestDreamBooth/GEN/gen_res.png'

    # reference_image_path = './examples/TestDreamBooth/FG/02.png'
    # bg_image_path = './examples/TestDreamBooth/BG/000000047948_GT.png'
    # bg_mask_path = './examples/TestDreamBooth/BG/000000047948_mask.png'
    # save_path = './examples/TestDreamBooth/GEN/gen_res2.png'

    img_list = ['4495', '11197', '11760', '15079', '17627', '21167', '63740', \
                '67180', '80666', '121031', '123213', '130599', '131131', '153299', \
                '198960', '210273', '212559', '218091', '236784', '252219', '321333', \
                '340175', '342006']

    # img_list = ['67180']
    # img_list = ['153299']
    reference_image_path = '/data/hang/customization/data/COCO/mask/val2017'
    bg_image_path = '/data/hang/customization/data/COCO/inpaint/val2017'
    source_image_path = '/data/hang/customization/data/COCO/val2017'
    bg_mask_path = reference_image_path
    image_dir = "/data/hang/customization/data/COCO"
    # save_path = 'results/COCO_inverse'
    # save_path = 'results/COCO'
    # save_path = 'results/COCO_retrain_67180'
    # save_path = 'results/COCO_batch_multiobj_obj365'
    # save_path = 'results/COCO_batch_multiobj_lvis_all'
    # save_path = 'results/COCO_batch_multiobj_obj365_ep8'
    # save_path = 'results/COCO_batch_multiobj_obj365_ep1_new_100k'
    save_path = 'results/LVIS_test5'
    # save_path = 'results/mask_dilated_LVIS'
    # save_path = 'results/COCO_batch_multiobj_obj365_ep1_new_100k_all_mask_all_real'
    os.makedirs(save_path, exist_ok=True)

    # Define paths to the dataset
    data_dir = '/data/hang/customization/data/COCO'  # Update this path
    set_split = 'val2017'
    # set_split = 'train2017'
    # ann_file = os.path.join(data_dir, 'annotations/instances_'+set_split+'.json')
    # ann_file = os.path.join(data_dir, 'lvis_v1/lvis_v1_'+set_split+'.json')
    set_split = 'val'
    ann_file = os.path.join(data_dir, 'lvis_v1/lvis_v1_'+set_split+'.json')
    img_dir = os.path.join(data_dir, set_split)

    obj_thr = 2

    # Initialize COCO API
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds()
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        if True:
        # if str(img_id) in img_list:
            # if True: 
            try:
                # print(img_id, flush=True)

                # obtain (inpainted) back image -> back.png
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
                anns = coco.loadAnns(ann_ids)

                # print(anns)
                anno_ = anns
                obj_ids = []
                obj_areas = []
                for i in range(len(anns)):
                    obj = anns[i]
                    area = obj['area']
                    if area > 3600:
                    # if area > 1000:
                        obj_ids.append(i)
                        obj_areas.append(area)

                sorted_obj_ids = np.argsort(obj_areas)[::-1]
                if len(sorted_obj_ids) >= obj_thr:
                    sorted_obj_ids = sorted_obj_ids[:obj_thr]

                if len(sorted_obj_ids)<obj_thr:
                    continue


                back_image = cv2.imread(os.path.join(bg_image_path, str(img_id)+'.png')).astype(np.uint8)
                back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)
                back_image_masked = back_image
                back_image_dir = os.path.join(save_path, 'back')
                print('back_image_dir:', back_image_dir)
                if not os.path.exists(back_image_dir):
                    os.mkdir(back_image_dir)
                # cv2.imwrite(os.path.join(back_image_dir, str(img_id)+'back.png'), cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB))
                os.makedirs('/data/hang/customization/data/LVIS_test/back/', exist_ok=True)
                cv2.imwrite('/data/hang/customization/data/LVIS_test/back/'+str(img_id)+'.jpg', cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB))

                source_image_dir = os.path.join(save_path, 'source')
                if not os.path.exists(source_image_dir):
                    os.mkdir(source_image_dir)
                source_image = cv2.imread(os.path.join(source_image_path, str(img_id).zfill(12)+'.jpg'))
                # print(os.path.join(source_image_path, str(img_id).zfill(12)+'.jpg'))
                # import pdb; pdb.set_trace()
                # cv2.imwrite(os.path.join(source_image_dir, str(img_id)+'source.png'), source_image)
                os.makedirs('/data/hang/customization/data/LVIS_test/source/', exist_ok=True)
                cv2.imwrite('/data/hang/customization/data/LVIS_test/source/'+str(img_id)+'.jpg', source_image)


                # for obj_id in sorted_obj_ids:
                #     anno_id = anns[obj_ids[obj_id]]
                #     ref_mask = self.lvis_api.ann_to_mask(anno_id)


                # import pdb; pdb.set_trace()

                # for ann in anns:
                counter = 0
                item_with_collage = {}

                for obj_id in sorted_obj_ids:
                    anno_id = anns[obj_ids[obj_id]]
                    category_id = anno_id['category_id']
                    bbox = anno_id['bbox']
                    # print('obj_ids: ', obj_ids)
                    # print('obj_id: ', obj_id)
                    # print(anno_id)

                    patch_dir = os.path.join(image_dir, 'patch_test', 'multiview', str(anno_id['id']))
                    patch_dir2 = "/data/hang/customization/data/LVIS_test/multiview/"+str(img_id)+'/'+str(counter)+'/'
                    os.makedirs(patch_dir2, exist_ok=True)
                    # print(patch_dir) # 495624
                    # print(category_id) # 1
                    # print(img_id) # 252219
                    # print(str(anno_id['id']))
                    # wwwwww

                    for file_name in os.listdir(patch_dir):
                        src_file = os.path.join(patch_dir, file_name)
                        dst_file = os.path.join(patch_dir2, file_name)

                        # 复制文件
                        if os.path.isfile(src_file):
                            shutil.copy(src_file, dst_file)


                    segmentation = anno_id['segmentation']
                    if isinstance(segmentation, list):  # Polygon 格式
                        tar_mask = np.zeros((500, 500), dtype=np.uint8)  # 假设图像大小为 500x500
                        polygons = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(tar_mask, [polygons], color=255)

                    elif isinstance(segmentation, dict):  # RLE 格式
                        tar_mask = mask_utils.decode(segmentation)  # 解码为二值掩码

                    else:
                        raise ValueError("Unsupported segmentation format.")

                    # 保存掩码图像
                    # mask_path = os.path.join(patch_dir, f"mask_{obj_id}.png")
                    # cv2.imwrite('test_massk.png', tar_mask)


                    # obtain object mask image -> mask.png
                    # bg_mask_path_i = os.path.join(bg_mask_path, str(category_id), str(img_id), str(anno_id['id']), "image_mask.png")
                    # print(bg_mask_path_i)
                    # # wwwwww
                    # tar_mask = cv2.imread(bg_mask_path_i)[:,:,0] > 128
                    # tar_mask = tar_mask.astype(np.uint8)
                    # mask_image_dir = os.path.join(save_path, 'mask')
                    # if not os.path.exists(mask_image_dir):
                    #     os.mkdir(mask_image_dir)
                    # cv2.imwrite(os.path.join(mask_image_dir, str(img_id)+str(anno_id['id'])+'mask.png'), tar_mask*255)

                    # back_image_masked[tar_mask == 1] = 255

                    # obtain square region of the mask image -> mask_.png
                    # output_image = np.zeros_like(tar_mask)
                    # contours, _ = cv2.findContours(tar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # # c = max(contours, key=cv2.contourArea)

                    # # Initialize the overall bounding box
                    # x_min, y_min = np.inf, np.inf
                    # x_max, y_max = -np.inf, -np.inf

                    # # Loop over each contour and update the overall bounding box
                    # for contour in contours:
                    #     x_, y_, w_, h_ = cv2.boundingRect(contour)
                    #     x_min = min(x_min, x_)
                    #     y_min = min(y_min, y_)
                    #     x_max = max(x_max, x_ + w_)
                    #     y_max = max(y_max, y_ + h_)

                    # # Final overall bounding box
                    # x = x_min
                    # y = y_min
                    # w = x_max - x_min
                    # h = y_max - y_min

                    # # x, y, w, h = cv2.boundingRect(c)
                    # # cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 255), 2) # output bounding box
                    # output_image[y:y+h, x:x+w] = 255
                    # mask__image_dir = os.path.join(save_path, 'box')
                    # if not os.path.exists(mask__image_dir):
                    #     os.mkdir(mask__image_dir)

                    x, y, w, h = map(int, bbox)
                    patch = source_image[y:y + h, x:x + w, :3]  # 裁剪原始 RGB 图像

                    tar_mask = np.zeros((back_image.shape[0], back_image.shape[1]))
                    tar_mask[y:y + h, x:x + w] = 255

                    # 生成 alpha 通道
                    mask = np.zeros(back_image.shape[:2], dtype=np.uint8)
                    if segmentation:
                        polygons = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [polygons], color=255)

                    # 裁剪 alpha 通道
                    alpha_patch = mask[y:y + h, x:x + w]

                    # 生成 RGBA 图像
                    patch = np.dstack((patch, alpha_patch))


                    # cv2.imwrite(os.path.join(mask__image_dir, str(img_id)+str(anno_id['id'])+'mask_.png'), output_image)
                    print('/data/hang/customization/data/LVIS_test/patch/'+str(img_id)+'/')
                    os.makedirs('/data/hang/customization/data/LVIS_test/patch/'+str(img_id)+'/', exist_ok=True)
                    cv2.imwrite('/data/hang/customization/data/LVIS_test/patch/'+str(img_id)+'/'+str(counter)+'.png', patch)
                    # cv2.imwrite('/data/hang/customization/data/LVIS_test/'+str(img_id)+'.jpg', cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB))
                    

                    back_image_masked[y:y+h, x:x+w] = 255

                    # obtain RGBA patch image -> patch.png
                    # patch_path_i = os.path.join(bg_mask_path, str(category_id), str(img_id), str(anno_id['id']), "patch.png")
                    # # image = cv2.imread(download_image(img_id), cv2.IMREAD_UNCHANGED)
                    # patch = cv2.imread(patch_path_i, cv2.IMREAD_UNCHANGED)
                    # patch_image_dir = os.path.join(save_path, 'patch')
                    # if not os.path.exists(patch_image_dir):
                    #     os.mkdir(patch_image_dir)
                    # cv2.imwrite(os.path.join(patch_image_dir, str(img_id)+str(anno_id['id'])+'patch.png'), patch)
                    print('/data/hang/customization/data/LVIS_test/bbx/'+str(img_id)+'/')
                    os.makedirs('/data/hang/customization/data/LVIS_test/bbx/'+str(img_id)+'/', exist_ok=True)
                    cv2.imwrite('/data/hang/customization/data/LVIS_test/bbx/'+str(img_id)+'/'+str(counter)+'.png', tar_mask)
                    mask = (patch[:,:,-1] > 128).astype(np.uint8)
                    image = patch[:,:,:-1]
                    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

                    # obtain item_with_collage
                    # item = process_pairs_multiple(image, mask, back_image.copy(), tar_mask, patch_dir, counter)
                    # item_with_collage.update(item)
                    counter += 1
                    
                    # import pdb; pdb.set_trace()
                    # ref_mask = self.lvis_api.ann_to_mask(anno_id)
                    # tar_image, tar_mask = ref_image.copy(), ref_mask.copy()

                # for ann in reversed(anns):
                #     category_id = ann['category_id']

                #     # obtain object mask image -> mask.png
                #     bg_mask_path_i = os.path.join(bg_mask_path, str(category_id), str(img_id), str(ann['id']), "image_mask.png")
                #     tar_mask = cv2.imread(bg_mask_path_i)[:,:,0] > 128
                #     tar_mask = tar_mask.astype(np.uint8)
                #     cv2.imwrite(os.path.join(save_path, str(img_id)+str(ann['id'])+'mask.png'), tar_mask*255)

                #     # obtain square region of the mask image -> mask_.png
                #     output_image = np.zeros_like(tar_mask)
                #     contours, _ = cv2.findContours(tar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #     c = max(contours, key=cv2.contourArea)
                #     x, y, w, h = cv2.boundingRect(c)
                #     # cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 255), 2) # output bounding box
                #     output_image[y:y+h, x:x+w] = 255
                #     cv2.imwrite(os.path.join(save_path, str(img_id)+str(ann['id'])+'mask_.png'), output_image)

                #     # obtain RGBA patch image -> patch.png
                #     patch_path_i = os.path.join(bg_mask_path, str(category_id), str(img_id), str(ann['id']), "patch.png")
                #     # image = cv2.imread(download_image(img_id), cv2.IMREAD_UNCHANGED)
                #     patch = cv2.imread(patch_path_i, cv2.IMREAD_UNCHANGED)
                #     cv2.imwrite(os.path.join(save_path, str(img_id)+str(ann['id'])+'patch.png'), patch)
                #     mask = (patch[:,:,-1] > 128).astype(np.uint8)
                #     image = patch[:,:,:-1]
                #     image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

                #     # obtain item_with_collage
                #     item = process_pairs_multiple(image, mask, back_image.copy(), tar_mask, counter)
                #     item_with_collage.update(item)
                #     counter += 1

                # print(item_with_collage)

                # # feed into a network
                # item_with_collage = process_composition(item_with_collage, obj_thr)
                # gen_image = inference_single_image_multi(item_with_collage, back_image)
                
                # comp_image_dir = os.path.join(save_path, 'composed')
                # if not os.path.exists(comp_image_dir):
                #     os.mkdir(comp_image_dir)
                # comp_save_path = os.path.join(comp_image_dir, 'composed'+str(img_id)+'.png')
                # cv2.imwrite(comp_save_path, gen_image[:,:,::-1])

                # back_image_masked_dir = os.path.join(save_path, 'back_box')
                # if not os.path.exists(back_image_masked_dir):
                #     os.mkdir(back_image_masked_dir)
                # back_masked_save_path = os.path.join(back_image_masked_dir, 'back_masked'+str(img_id)+'.png')
                # cv2.imwrite(back_masked_save_path, back_image_masked[:,:,::-1])
                # h,w = back_image.shape[0], back_image.shape[0]
                # ref_image = cv2.resize(ref_image, (w,h))
                # vis_image = cv2.hconcat([ref_image, back_image, gen_image])

                # comp_save_path = os.path.join(save_path, str(img_id)+'.png')
                # cv2.imwrite(comp_save_path, back_image[:,:,::-1])
                







                

                # bbox_path = os.path.join("output2.txt")
                # with open(bbox_path, 'a+') as f:
                #     f.write(str(img_id)+'\n')
                #     f.write(str(np.shape(back_image)))
                #     f.write(str(np.shape(tar_mask)))
                #     f.write(str(np.shape(image)))
                #     f.write(str(np.shape(mask)))
            except: 
                print('fail')
                
                

    



    # reference image + reference mask
    # You could use the demo of SAM to extract RGB-A image with masks
    # https://segment-anything.com/demo
    # image = cv2.imread( reference_image_path, cv2.IMREAD_UNCHANGED)
    # mask = (image[:,:,-1] > 128).astype(np.uint8)
    # image = image[:,:,:-1]
    # image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    # ref_image = image 
    # ref_mask = mask

    # # background image
    # back_image = cv2.imread(bg_image_path).astype(np.uint8)
    # back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

    # # background mask 
    # tar_mask = cv2.imread(bg_mask_path)[:,:,0] > 128
    # tar_mask = tar_mask.astype(np.uint8)
    
    # gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
    # h,w = back_image.shape[0], back_image.shape[0]
    # ref_image = cv2.resize(ref_image, (w,h))
    # vis_image = cv2.hconcat([ref_image, back_image, gen_image])
    
    # cv2.imwrite(save_path, vis_image [:,:,::-1])






    # '''
    '''
    # ==== Example for inferring VITON-HD Test dataset ===

    from omegaconf import OmegaConf
    import os 
    DConf = OmegaConf.load('./configs/datasets.yaml')
    save_dir = './VITONGEN'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    test_dir = DConf.Test.VitonHDTest.image_dir
    image_names = os.listdir(test_dir)
    
    for image_name in image_names:
        ref_image_path = os.path.join(test_dir, image_name)
        tar_image_path = ref_image_path.replace('/cloth/', '/image/')
        ref_mask_path = ref_image_path.replace('/cloth/','/cloth-mask/')
        tar_mask_path = ref_image_path.replace('/cloth/', '/image-parse-v3/').replace('.jpg','.png')

        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        gt_image = cv2.imread(tar_image_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

        tar_mask = Image.open(tar_mask_path ).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = tar_mask == 5

        gen_image = inference_single_image(ref_image, ref_mask, gt_image.copy(), tar_mask)
        gen_path = os.path.join(save_dir, image_name)

        vis_image = cv2.hconcat([ref_image, gt_image, gen_image])
        cv2.imwrite(gen_path, vis_image[:,:,::-1])
    '''

    

