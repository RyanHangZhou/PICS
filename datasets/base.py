import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from .data_utils import * 

class BaseDataset(Dataset):
    def __init__(self):
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

    def aug_data_mask(self, image, mask):
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT),
            ])

        transformed = transform(image=image.astype(np.uint8), mask=mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask

    # def aug_patch(self, patch):
    #     transform = A.Compose([
    #         A.HorizontalFlip(p=0.2),
    #         A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    #         A.Rotate(limit=15, border_mode=cv2.BORDER_REPLICATE, p=0.5),
    #         ])

    #     return transform(image=patch)["image"]

    def aug_patch(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        mask = (gray < 250).astype(np.float32)[:, :, None] 

        transform = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        ])

        transformed = transform(image=patch.astype(np.uint8), mask=mask)
        aug_img = transformed["image"]
        aug_mask = transformed["mask"]
        final_img = aug_img * aug_mask + 255 * (1 - aug_mask)

        return final_img.astype(np.uint8)

    def sample_timestep(self, max_step=1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0, max_step)
        else:
            step = np.random.randint(0, max_step // 2)
        return np.array([step])

    def get_patch(self, ref_image, ref_mask):
        '''
        extract compact patch and convert to 224x224 RGBA. 
        ref_mask: [0, 1]
        '''

        # 1. Get the outline Box of the reference image
        y1, y2, x1, x2 = get_bbox_from_mask(ref_mask) # y1y2x1x2, obtain location from ref patch
        
        # 2. Background is set to white (255)
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

        # 3. Crop based on bounding boxes
        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        ref_mask_crop = ref_mask[y1:y2, x1:x2] # obtain a tight mask

        # 4. Dilate the patch and mask
        ratio = np.random.randint(11, 15) / 10
        masked_ref_image, ref_mask_crop = expand_image_mask(masked_ref_image, ref_mask_crop, ratio=ratio)

        # augmentation
        # masked_ref_image, ref_mask_crop = self.aug_data_mask(masked_ref_image, ref_mask_crop) 

        # 5. Padding & Resize 
        masked_ref_image = pad_to_square(masked_ref_image, pad_value=255)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224, 224))

        m_local = ref_mask_crop[:, :, None] * 255
        m_local = pad_to_square(m_local, pad_value=0)
        m_local = cv2.resize(m_local.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
        
        rgba_image = np.dstack((masked_ref_image.astype(np.uint8), m_local))

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
        object_0 = object_0 / 255 
        item.update({'ref0': object_0.copy()}) # patch 0 (checked) [0, 1], 224x224x3

        ratio = np.random.randint(11, 15) / 10 
        object_1 = expand_image(object_1, ratio=ratio)
        object_1 = self.aug_patch(object_1)
        object_1 = pad_to_square(object_1, pad_value = 255, random = False) # pad to square
        object_1 = cv2.resize(object_1.astype(np.uint8), (224,224) ).astype(np.uint8) # check 1
        object_1 = object_1 / 255 
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
        item.update({'hint': background.copy()})

        item.update({'mask0': background_mask0_.copy()})
        item.update({'mask1': background_mask1_.copy()})

        sampled_time_steps = self.sample_timestep()
        item['time_steps'] = sampled_time_steps
        item['object_num'] = 2

        return item
