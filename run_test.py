import cv2
import os
import einops
import numpy as np
import torch
import argparse
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
from omegaconf import OmegaConf
from tqdm import tqdm

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

def get_input(batch, k):
    x = batch[k]
    if len(x.shape) == 3:
        x = x[None, ...]

    x = torch.tensor(x)
    x = einops.rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()
    return x

def get_unconditional_conditioning(N, obj_thr):
    x = [torch.zeros((1, 3, 224, 224)).to(model.device)] * N
    single_uc = model.get_learned_conditioning(x)
    uc = single_uc.unsqueeze(-1).repeat(1, 1, 1, obj_thr)
    return {"pch_code": uc}

def inference(item, back_image):
    obj_thr = 2
    num_samples = 1
    H, W = 512, 512
    guidance_scale = 5.0
    
    # 1. Condition & Mask Extraction
    xc = []
    xc_mask = []
    for i in range(obj_thr):
        xc.append(get_input(item, f"view{i}").cuda())
        xc_mask.append(get_input(item, f"mask{i}"))

    # 2. Cross-Attention Condition (pch_code)
    c_list = [model.get_learned_conditioning(xc_i) for xc_i in xc]
    c_tensor = torch.stack(c_list).permute(1, 2, 3, 0) # [B, Tokens, Dim, Obj]
    cond_cross = {"pch_code": c_tensor}

    # 3. Mask Condition
    c_mask = torch.stack(xc_mask).permute(1, 2, 3, 4, 0) # Align with BasicTransformerBlock

    # 4. ControlNet / Concat Condition
    hint = item['hint']
    control = torch.from_numpy(hint.copy()).float().cuda()
    control = torch.stack([control] * num_samples, dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    # 5. Build Final Condition Dictionaries
    cond = {
        "c_concat": [control], 
        "c_crossattn": [cond_cross], 
        "c_mask": [c_mask]
    }
    
    # Correctly unwrap the UC dictionary
    uc_pch = get_unconditional_conditioning(num_samples, obj_thr)
    un_cond = {
        "c_concat": [control], 
        "c_crossattn": [uc_pch], 
        "c_mask": [c_mask]
    }

    # 6. Sampling
    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    shape = (4, H // 8, W // 8)
    model.control_scales = [1.0] * 13
    
    samples, _ = ddim_sampler.sample(
        50, num_samples, shape, cond, 
        verbose=False, eta=0.0,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=un_cond
    )

    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    # 7. Post-processing
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()
    
    pred = np.clip(x_samples[0], 0, 255).astype(np.uint8)
    
    # Resize and crop
    side = max(back_image.shape[0], back_image.shape[1])
    pred = cv2.resize(pred, (side, side))
    pred = crop_back(pred, back_image, item['extra_sizes'], item['hint_sizes0'], item['hint_sizes1']) 
    
    return pred


def process_pairs_multiple(mask, tar_image, patch_dir, counter=0, max_ratio=0.8):
    # 1. Process Reference Object (View)
    view = cv2.imread(patch_dir)
    view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    view = expand_image(view)
    view = pad_to_square(view, pad_value=255, random=False)
    view = cv2.resize(view.astype(np.uint8), (224, 224))
    view = view.astype(np.float32) / 255.0

    # 2. BBox and Mask Logic
    box_yyxx = get_bbox_from_mask(mask)
    box_yyxx = expand_bbox(mask, box_yyxx, ratio=[1.1, 1.2])
    
    # Define crop area (using full image here)
    H1, W1 = tar_image.shape[0], tar_image.shape[1]
    box_yyxx_crop = [0, H1, 0, W1]
    
    # Handle box within crop
    y1, y2, x1, x2 = box_in_box(box_yyxx, box_yyxx_crop)

    # 3. Create Collage (Input Hint)
    # Background with hole (zeroed out at object position)
    collage = tar_image.copy()
    source_collage = collage.copy()
    collage[y1:y2, x1:x2, :] = 0

    # Binary mask for the current object hole
    collage_mask = np.zeros_like(tar_image, dtype=np.float32)
    collage_mask[y1:y2, x1:x2, :] = 1.0

    # 4. Square Padding & Resizing
    # Pad all to square (pad_value 2 for mask indicates padding area)
    tar_square = pad_to_square(tar_image, pad_value=0, random=False)
    collage_square = pad_to_square(collage, pad_value=0, random=False)
    mask_square = pad_to_square(collage_mask, pad_value=2, random=False)
    
    H2, W2 = collage_square.shape[0], collage_square.shape[1]

    # Resize to model input size
    tar_res = cv2.resize(tar_square, (512, 512)).astype(np.float32)
    col_res = cv2.resize(collage_square, (512, 512)).astype(np.float32)
    mask_res = cv2.resize(mask_square, (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    # 5. Mask Value Normalization
    # Original logic: mask=1 for object, 0 for background, -1 for padding
    mask_res[mask_res == 2] = -1
    
    # For conditioning: keep a 0/1 version for cross-attn mask
    c_mask = np.where(mask_res[..., 0:1] == 1, 1.0, 0.0).astype(np.float32)

    # 6. Final Item Assembly
    # Normalize images to [-1, 1]
    tar_res = tar_res / 127.5 - 1.0
    col_res = col_res / 127.5 - 1.0
    
    # Hint: Concatenate background with the (-1, 0, 1) mask
    hint_final = np.concatenate([col_res, mask_res[..., :1]], axis=-1)

    item = {
        f'view{counter}': view,
        f'hint{counter}': hint_final,
        f'mask{counter}': c_mask,
        f'hint_sizes{counter}': np.array([y1, x1, y2, x2]),
        'jpg': tar_res, # Targets are same for all counters in a pair
        'collage': source_collage,
        'extra_sizes': np.array([H1, W1, H2, W2])
    }

    return item

def process_composition(item, obj_thr):
    collage = item['collage']
    collage_mask = collage.copy() * 0.0

    for i in reversed(range(obj_thr)):
        y1, x1, y2, x2 = item['hint_sizes'+str(i)]
        collage[y1:y2, x1:x2, :] = 0
        collage_mask[y1:y2,x1:x2,:] = 1.0

    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
    collage = collage / 127.5 - 1.0 

    collage_mask = pad_to_square(collage_mask, pad_value = 0, random = False).astype(np.uint8)
    collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512, 512),  interpolation=cv2.INTER_NEAREST).astype(np.float32)

    collage = np.concatenate([collage, collage_mask[:,:,:1]] , -1)
    item.update({'hint': collage.copy()})
    return item


# if __name__ == '__main__': 
#     sample_num = 31
#     compose_num = 2
#     obj_thr = 2
#     save_path = 'results/LVIS_again2'
#     input_path = '/data/hang/customization/data/Wild'
#     os.makedirs(save_path, exist_ok=True)
#     img_list = os.listdir(input_path)

#     for i in img_list:
#         print(os.path.join(input_path, i+'/'+'image.jpg'))
#         back_image = cv2.imread(os.path.join(input_path, i+'/'+'image.jpg')).astype(np.uint8)
#         back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

#         counter = 0
#         item_with_collage = {}
#         for j in range(compose_num):
#             patch_dir = os.path.join(input_path, i+"/object_"+str(j)+".png")
#             bg_mask_path_i = os.path.join(input_path, i+"/object_"+str(j)+"_mask.png")
#             tar_mask = cv2.imread(bg_mask_path_i)[:,:,0] > 128
#             tar_mask = tar_mask.astype(np.uint8)

#             item = process_pairs_multiple(tar_mask, back_image.copy(), patch_dir, counter)
#             item_with_collage.update(item)
#             counter += 1

#         item_with_collage = process_composition(item_with_collage, obj_thr)
#         gen_image = inference(item_with_collage, back_image)
        
#         comp_image_dir = os.path.join(save_path, 'composed')
#         if not os.path.exists(comp_image_dir):
#             os.mkdir(comp_image_dir)
#         comp_save_path = os.path.join(comp_image_dir, 'composed'+str(i)+'__'+'.png')
#         cv2.imwrite(comp_save_path, gen_image[:,:,::-1])


def run_inference(input_dir, output_dir, sample_num=31, obj_thr=2):
    """
    Core inference loop for multi-object composition.
    """
    os.makedirs(output_dir, exist_ok=True)
    comp_image_dir = os.path.join(output_dir, 'composed')
    os.makedirs(comp_image_dir, exist_ok=True)

    img_ids = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for img_id in tqdm(img_ids, desc="Processing images"):
        img_folder = os.path.join(input_dir, img_id)
        img_path = os.path.join(img_folder, 'image.jpg')
        
        if not os.path.exists(img_path):
            continue

        # 1. Load background image
        back_image = cv2.imread(img_path)
        back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

        # 2. Iteratively process multiple objects
        item_with_collage = {}
        for j in range(obj_thr):
            patch_path = os.path.join(img_folder, f"object_{j}.png")
            mask_path = os.path.join(img_folder, f"object_{j}_mask.png")
            
            if not (os.path.exists(patch_path) and os.path.exists(mask_path)):
                print(f"Warning: Object {j} missing in {img_id}")
                continue

            tar_mask = (cv2.imread(mask_path)[:, :, 0] > 128).astype(np.uint8)
            
            # Pass counter=j to ensure keys like 'view0', 'view1' are unique
            item = process_pairs_multiple(tar_mask, back_image, patch_path, counter=j)
            item_with_collage.update(item)

        # 3. Composition & Model Prediction
        # Ensure process_composition merges 'hint0', 'hint1' into a single 'hint'
        item_with_collage = process_composition(item_with_collage, obj_thr)
        
        # Using inference_single_image_multi as defined previously
        gen_image = inference(item_with_collage, back_image)
        
        # 4. Save result
        save_name = f'composed_{img_id}.png'
        cv2.imwrite(os.path.join(comp_image_dir, save_name), gen_image[:, :, ::-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input data directory')
    parser.add_argument('--output', type=str, help='Output save directory')
    parser.add_argument('--obj_thr', type=int, default=2, help='Number of objects to compose')
    args = parser.parse_args()
    
    run_inference(args.input, args.output, obj_thr=args.obj_thr)