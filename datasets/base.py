import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A


class BaseDataset(Dataset):
    def __init__(self):
        image_mask_dict = {}
        self.data = []

    def __getitem__(self, idx):
        item = self._get_sample(idx)
        return item
                
    def _get_sample(self, idx):
        # Implemented for each specific dataset
        pass

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        pass

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag

    def aug_data_mask(self, image, mask):
        transform = A.Compose([
            # A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT),
            ])

        transformed = transform(image=image.astype(np.uint8), mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask

    def aug_patch(self, patch):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE),
            ])

        transformed = transform(image=patch)
        transformed_patch = transformed["image"]
        return transformed_patch

    def sample_timestep(self, max_step = 1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0,max_step)
            return np.array([step])
        step_start = 0
        step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def get_patch(self, ref_image, ref_mask):
        '''
        ref_mask: [0, 1]
        '''

        # 1. Get the outline Box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask) # y1y2x1x2, obtain location from ref patch
        # assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
        
        # 2. Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) # obtain patch (outside white) [0, 255]

        # 3. Crop based on bounding boxes
        y1,y2,x1,x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
        ref_mask = ref_mask[y1:y2,x1:x2] # obtain a tight mask

        # 4. Dilate the patch and mask
        # ratio = 1 # np.random.randint(11, 15) / 10 
        ratio = np.random.randint(11, 15) / 10
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)

        # augmentation
        # masked_ref_image, ref_mask = self.aug_data_mask(masked_ref_image, ref_mask) 

        # 5. Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) # pad to square
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)

        rgba_image = np.dstack((masked_ref_image, ref_mask_3[:, :, 0]))
        return rgba_image

    def _construct_collage(self, image, object_0, object_1, mask_0, mask_1):
        background = image.copy()
        image = pad_to_square(image, pad_value = 0, random = False).astype(np.uint8)
        image = cv2.resize(image.astype(np.uint8), (512,512)).astype(np.float32)
        image = image / 127.5 - 1.0
        item = {}
        item.update({'jpg': image.copy()}) # source image (checked) [-1, 1], 512x512x3

        ratio = np.random.randint(11, 15) / 10 
        object_0 = expand_image(object_0, ratio=ratio)
        object_0 = self.aug_patch(object_0)
        object_0 = pad_to_square(object_0, pad_value = 255, random = False) # pad to square
        object_0 = cv2.resize(object_0.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1
        object_0 = object_0  / 255 
        item.update({'ref0': object_0.copy()}) # patch 0 (checked) [0, 1], 224x224x3

        ratio = np.random.randint(11, 15) / 10 
        object_1 = expand_image(object_1, ratio=ratio)
        object_1 = self.aug_patch(object_1)
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

        box_yyxx = get_bbox_from_mask(mask_1)
        box_yyxx = expand_bbox(mask_1, box_yyxx, ratio=[1.1, 1.2]) #1.1  1.3
        y1, y2, x1, x2 = box_yyxx
        background[y1:y2, x1:x2,:] = 0
        background_mask1[y1:y2, x1:x2, :] = 1.0
        background_mask[y1:y2, x1:x2, :] = 1.0

        background = pad_to_square(background, pad_value = 0, random = False).astype(np.uint8)
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

        item.update({'mask0': background_mask0_.copy()}) # mask (checked) [0, 1], 512x512
        item.update({'mask1': background_mask1_.copy()}) # mask (checked) [0, 1], 512x512

        sampled_time_steps = self.sample_timestep()
        item['time_steps'] = sampled_time_steps
        item['object_num'] = 2

        return item
