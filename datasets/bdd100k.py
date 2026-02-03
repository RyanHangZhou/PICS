import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from .data_utils import * 
from .base import BaseDataset
from lvis import LVIS
from scipy.ndimage import label
import PIL.ImageDraw as ImageDraw
from util.box_ops import mask_to_bbox_xywh, compute_iou_matrix, draw_bboxes
from util.cityscapes_ops import Annotation, name2label
from pathlib import Path
from pycocotools import mask as mask_utils
import shutil

IS_VERIFY = False

class BDD100KDataset(BaseDataset):
    def __init__(self, construct_dataset_dir, obj_thr=20, area_ratio=0.02):
        self.obj_thr = obj_thr
        self.construct_dataset_dir = construct_dataset_dir
        os.makedirs(Path(self.construct_dataset_dir), exist_ok=True)
        self.area_ratio = area_ratio
        self.sample_list = os.listdir(self.construct_dataset_dir)

    def _intersect_2_obj(self, image_dir, samples, idx):
        self.image_dir = image_dir
        sample = samples[idx]
        image_name = sample['name']
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        h, w = image.shape[0:2]
        image_area = h * w

        labels = sample['labels']

        # filter by area
        obj_ids = []
        obj_areas = []
        obj_bbox = []
        for i in range(len(labels)):
            obj = labels[i]
            bbox = [obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']]
            rle = obj['rle']
            mask = mask_utils.decode(rle)
            area = np.sum(mask)
            if area > image_area * self.area_ratio:
                obj_ids.append(i)
                obj_areas.append(area)
                obj_bbox.append(bbox)

        if len(obj_bbox) < 2:
            print(f"[Info] Skip image index {image_name[:-4]} due to insufficient bbox.")
            return

        os.makedirs(Path(self.construct_dataset_dir) / image_name[:-4], exist_ok=True)
        bbox_xyxy = np.array(obj_bbox)

        if IS_VERIFY:
            image_with_boxes = draw_bboxes(image, bbox_xyxy)
            cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name[:-4] / "bboxes_image.png"), image_with_boxes)
        
        iou_matrix = compute_iou_matrix(bbox_xyxy)
        np.fill_diagonal(iou_matrix, -1) # Exclude self-comparisons (i.e., each box with itself)

        max_index = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        index0, index1 = max_index[0], max_index[1]
        max_iou = iou_matrix[index0, index1]

        if max_iou <= 0:
            print(f"[Info] Skip image index {image_name[:-4]} due to no overlapping bboxes.")
            return

        dst = Path(self.construct_dataset_dir) / image_name[:-4] / "image.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, dst)

        box0 = obj_bbox[index0]
        box1 = obj_bbox[index1]

        counter = 0
        for i in range(len(labels)):
            obj = labels[i]
            rle = obj['rle']
            if counter == obj_ids[index0]:
                mask = mask_utils.decode(rle)
            counter += 1
        cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name[:-4] / "object_0_mask.png"), 255*mask)
        patch = self.get_patch(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask)
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name[:-4] / "object_0.png"), patch)

        if IS_VERIFY:
            mask_color = np.stack([mask * 255]*3, axis=-1).astype(np.uint8)
            highlight = np.zeros_like(image)
            highlight[:, :, 2] = 255  # red channel
            alpha = 0.5
            image_with_boxes = np.where(mask_color == 255, cv2.addWeighted(image_with_boxes, 1 - alpha, highlight, alpha, 0), image_with_boxes)

        counter = 0
        for i in range(len(labels)):
            obj = labels[i]
            rle = obj['rle']
            if counter == obj_ids[index1]:
                mask = mask_utils.decode(rle)
            counter += 1
        cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name[:-4] / "object_1_mask.png"), 255*mask)
        patch = self.get_patch(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask)
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name[:-4] / "object_1.png"), patch)

        if IS_VERIFY:
            mask_color = np.stack([mask * 255]*3, axis=-1).astype(np.uint8)
            highlight = np.zeros_like(image)
            highlight[:, :, 0] = 255  # blue channel
            alpha = 0.5
            image_with_boxes = np.where(mask_color == 255, cv2.addWeighted(image_with_boxes, 1 - alpha, highlight, alpha, 0), image_with_boxes)
            cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name[:-4] / "highlighted_image.png"), image_with_boxes)

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
    two-object case: train/test: 1012/371
    '''
    import argparse

    parser = argparse.ArgumentParser(description="BDD100KDataset Analysis")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--construct_dataset_dir", type=str, default='bin', help="Path to the debug bin directory.")
    parser.add_argument("--dataset_name", type=str, default='bdd100k', help="Dataset name.")
    parser.add_argument('--is_train', action='store_true', help="Train/Test")
    parser.add_argument('--is_build_data', action='store_true', help="Build data")
    parser.add_argument('--is_multiple', action='store_true', help="Multiple/Two objects")
    parser.add_argument("--area_ratio", type=float, default=0.01171, help="Area ratio for filtering out small objects.")
    parser.add_argument("--obj_thr", type=int, default=20, help="Object threshold for filtering.")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to test.")
    args = parser.parse_args()

    if args.is_train:
        image_dir = Path(args.dataset_dir) / args.dataset_name / "images" / "10k" / "train"
        json_path = Path(args.dataset_dir) / args.dataset_name / "labels" / "ins_seg" / "rles" / "ins_seg_train.json"
        max_num = 7000
    else:
        image_dir = Path(args.dataset_dir) / args.dataset_name / "images" / "10k" / "val"
        json_path = Path(args.dataset_dir) / args.dataset_name / "labels" / "ins_seg" / "rles" / "ins_seg_val.json"
        max_num = 1000

    dataset = BDD100KDataset(
        construct_dataset_dir = args.construct_dataset_dir,
        obj_thr = args.obj_thr, 
        area_ratio = args.area_ratio, 
    )

    with open(json_path) as data_file:
        label = json.load(data_file)
    samples = label["frames"]

    if args.is_build_data: 
        if not args.is_multiple:
            for index in range(max_num):
                dataset._intersect_2_obj(image_dir, samples, index)
    else:
        for index in range(len(os.listdir(args.construct_dataset_dir))):
            collage = dataset._get_sample(index)
