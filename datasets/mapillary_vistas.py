import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from .data_utils import * 
from .base import BaseDataset
from scipy.ndimage import label
from util.box_ops import mask_to_bbox_xywh, compute_iou_matrix, draw_bboxes
from pathlib import Path
import shutil

IS_VERIFY = False

class MapillaryVistasDataset(BaseDataset):
    def __init__(self, construct_dataset_dir, obj_thr=20, area_ratio=0.02):
        self.obj_thr = obj_thr
        self.construct_dataset_dir = construct_dataset_dir
        os.makedirs(Path(self.construct_dataset_dir), exist_ok=True)
        self.area_ratio = area_ratio
        self.sample_list = os.listdir(self.construct_dataset_dir)

    def _intersect_2_obj(self, image_dir, instance_dir, labels, idx):
        json_list = os.listdir(instance_dir)
        image_name = json_list[idx][:-4]
        image_path = os.path.join(image_dir, image_name+'.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instance_path = os.path.join(instance_dir, image_name+'.png')
        instance_image = Image.open(instance_path)
        instance_array = np.array(instance_image, dtype=np.uint16)

        instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
        instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)

        img_h, img_w = image.shape[0:2]
        image_area = img_h*img_w

        # vehicle_keywords = ['car', 'truck', 'bus']
        # excluded_keywords = ['bicycle']

        # vehicle_ids = []
        # for idx, label in enumerate(labels):
        #     name = label['name'].lower()
        #     if any(k in name for k in vehicle_keywords) and not any(k in name for k in excluded_keywords):
        #         vehicle_ids.append(idx)

        '''
        ids: 107, 'name': 'object--vehicle--bus', 'readable': 'Bus', 'color': [0, 60, 100]
        ids: 108, 'name': 'object--vehicle--car', 'readable': 'Car', 'color': [0, 0, 142]
        ids: 109, 'name': 'object--vehicle--caravan', 'readable': 'Caravan', 'color': [0, 0, 90]
        ids: 114, 'name': 'object--vehicle--truck', 'readable': 'Truck', 'color': [0, 0, 70]
        '''

        target_class_ids = [107, 108, 109, 114]
        max_instance = np.max(instance_ids_array)

        obj_ids = []
        obj_areas = []
        obj_bbox = []
        counter = 0
        for target_id in target_class_ids:
            semantic_mask = (instance_label_array == target_id)
            for idx in range(max_instance):
                instance_mask = (instance_ids_array == idx)
                mask = np.logical_and(semantic_mask, instance_mask).astype(np.uint8)
                area = np.sum(mask)
                bbox = mask_to_bbox_xywh(mask)
                if area > image_area * self.area_ratio:
                    obj_ids.append(counter)
                    obj_areas.append(area)
                    obj_bbox.append(bbox)
                counter += 1

        if len(obj_bbox) < 2:
            print(f"[Info] Skip image index {image_name} due to insufficient bbox.")
            return

        # filter by IOU
        bbox_xyxy = []
        for box in obj_bbox:
            x, y, w, h = box
            bbox_xyxy.append([x, y, x + w, y + h])
        bbox_xyxy = np.array(bbox_xyxy)  # shape: [N, 4]
        os.makedirs(Path(self.construct_dataset_dir) / image_name, exist_ok=True)

        if IS_VERIFY:
            image_with_boxes = draw_bboxes(image, bbox_xyxy)
            cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name / "bboxes_image.png"), cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))


        iou_matrix = compute_iou_matrix(bbox_xyxy)
        np.fill_diagonal(iou_matrix, -1) # Exclude self-comparisons (i.e., each box with itself)

        max_index = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        index0, index1 = max_index[0], max_index[1]
        max_iou = iou_matrix[index0, index1]

        if max_iou <= 0:
            print(f"[Info] Skip image index {image_name} due to no overlapping bboxes.")
            return

        dst = Path(self.construct_dataset_dir) / image_name / "image.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, dst)

        counter = 0
        found = False
        for target_id in target_class_ids:
            semantic_mask = (instance_label_array == target_id)
            for idx in range(max_instance):
                if counter == obj_ids[index0]:
                    instance_mask = (instance_ids_array == idx)
                    mask = np.logical_and(semantic_mask, instance_mask).astype(np.uint8)
                    found = True
                    break
                counter += 1
            if found:
                break
        cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name / "object_0_mask.png"), 255*mask)
        patch = self.get_patch(image, mask)
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name / "object_0.png"), patch)

        if IS_VERIFY:
            mask_color = np.stack([mask * 255]*3, axis=-1).astype(np.uint8)
            highlight = np.zeros_like(image)
            highlight[:, :, 2] = 255  # red channel
            alpha = 0.5
            image_with_boxes = np.where(mask_color == 255, cv2.addWeighted(image_with_boxes, 1 - alpha, highlight, alpha, 0), image_with_boxes)

        counter = 0
        found = False
        for target_id in target_class_ids:
            semantic_mask = (instance_label_array == target_id)
            for idx in range(max_instance):
                if counter == obj_ids[index1]:
                    instance_mask = (instance_ids_array == idx)
                    mask = np.logical_and(semantic_mask, instance_mask).astype(np.uint8)
                    found = True
                    break
                counter += 1
            if found:
                break

        cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name / "object_1_mask.png"), 255*mask)
        patch = self.get_patch(image, mask)
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name / "object_1.png"), patch)

        if IS_VERIFY:
            mask_color = np.stack([mask * 255]*3, axis=-1).astype(np.uint8)
            highlight = np.zeros_like(image)
            highlight[:, :, 0] = 255  # blue channel
            alpha = 0.5
            image_with_boxes = np.where(mask_color == 255, cv2.addWeighted(image_with_boxes, 1 - alpha, highlight, alpha, 0), image_with_boxes)
            cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name / "highlighted_image.png"), cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

    def _get_sample(self, idx):
        sample_path = os.path.join(self.construct_dataset_dir, self.sample_list[idx])
        image = cv2.cvtColor(cv2.imread(os.path.join(sample_path, "image.jpg")), cv2.COLOR_BGR2RGB)
        object_0 = cv2.cvtColor(cv2.imread(os.path.join(sample_path, "object_0.png")), cv2.COLOR_BGR2RGB)
        object_1 = cv2.cvtColor(cv2.imread(os.path.join(sample_path, "object_1.png")), cv2.COLOR_BGR2RGB)
        mask_0 = cv2.imread(os.path.join(sample_path, "object_0_mask.png"), cv2.IMREAD_GRAYSCALE)
        mask_1 = cv2.imread(os.path.join(sample_path, "object_1_mask.png"), cv2.IMREAD_GRAYSCALE)
        collage = self._construct_collage(image, object_0, object_1, mask_0, mask_1)
        return collage

    def __len__(self):
        return len(os.listdir(self.construct_dataset_dir))


if __name__ == "__main__":
    '''
    two-object case: train/test: 603/190
    '''
    import argparse

    parser = argparse.ArgumentParser(description="MapillaryVistasDataset Analysis")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--construct_dataset_dir", type=str, default='bin', help="Path to the debug bin directory.")
    parser.add_argument("--dataset_name", type=str, default='MVD', help="Dataset name.")
    parser.add_argument('--is_train', action='store_true', help="Train/Test")
    parser.add_argument('--is_build_data', action='store_true', help="Build data")
    parser.add_argument('--is_multiple', action='store_true', help="Multiple/Two objects")
    parser.add_argument("--area_ratio", type=float, default=0.01171, help="Area ratio for filtering out small objects.")
    parser.add_argument("--obj_thr", type=int, default=20, help="Object threshold for filtering.")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to test.")
    args = parser.parse_args()

    version = "v2.0" # "v1.2"
    config_path = Path(args.dataset_dir) / args.dataset_name / f'config_{version}.json'
    with open(config_path) as config_file:
        config = json.load(config_file)
    labels = config['labels']

    if args.is_train:
        image_dir = Path(args.dataset_dir) / args.dataset_name / "training" / "images"
        instance_dir = Path(args.dataset_dir) / args.dataset_name / "training" / "v2.0" / "instances"
        max_num = 18000
    else:
        image_dir = Path(args.dataset_dir) / args.dataset_name / "validation" / "images"
        instance_dir = Path(args.dataset_dir) / args.dataset_name / "validation" / "v2.0" / "instances"
        max_num = 2000

    dataset = MapillaryVistasDataset(
        construct_dataset_dir = args.construct_dataset_dir,
        obj_thr = args.obj_thr, 
        area_ratio = args.area_ratio, 
    )

    if args.is_build_data: 
        if not args.is_multiple:
            for index in range(max_num):
                dataset._intersect_2_obj(image_dir, instance_dir, labels, index)
                print('Done index ', index)
    else:
        for index in range(len(os.listdir(args.construct_dataset_dir))):
            collage = dataset._get_sample(index)


'''
25,000 high-resolution images
124 semantic object categories
100 instance-specifically annotated categories
Global reach, covering 6 continents
Variety of weather, season, time of day, camera, and viewpoint
'''      
