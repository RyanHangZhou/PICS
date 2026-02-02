import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
from pycocotools import mask as mask_utils
# from lvis import LVIS
from pycocotools.coco import COCO
from pathlib import Path
from util.box_ops import compute_iou_matrix, draw_bboxes
import shutil

IS_VERIFY = True


class Objects365Dataset(BaseDataset):
    def __init__(self, json_dir, construct_dataset_dir, obj_thr=20, area_ratio=0.02):
        self.obj_thr = obj_thr
        self.dynamic = 0
        self.json_dir = json_dir
        self.construct_dataset_dir = construct_dataset_dir
        self.area_ratio = area_ratio
        self.json_list = self._get_all_file_paths_recursive(json_dir)

    def _get_all_file_paths_recursive(self, root_dir):
        all_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                abs_path = os.path.abspath(os.path.join(dirpath, f))
                all_files.append(abs_path)
        return all_files

    def _get_image_path(self, file_name):
        for img_dir in self.image_dir:
            path = img_dir / file_name
            if path.exists():
                return str(path)
        raise FileNotFoundError(f"File {file_name} not found in any of the image_dir.")

    def _intersect_2_obj(self, image_dir, json_dir, idx):
        self.image_dir = image_dir
        json_path = self.json_list[idx]
        image_name = json_path.split('/')[-1]
        image_subset = json_path.split('/')[-2]

        image_path = os.path.join(os.path.join(image_dir, image_subset), image_name[:-5]+'.jpg')
        print('image_path', image_path)
        image = cv2.imread(image_path)


        with open(json_path) as f:
            data = json.load(f)
            image_id = data["image_id"]
            annotations = data["annotations"]

        # lvis_api = LVIS(json_path)
        # img_ids = sorted(lvis_api.imgs.keys())
        # imgs_info = lvis_api.load_imgs(img_ids)
        # annos = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

        h, w = image.shape[0:2]
        image_area = h*w

        anno = annotations

        # filter by area
        obj_ids = []
        obj_areas = []
        obj_bbox = []
        for i in range(len(anno)):
            obj = anno[i]
            area = obj['area']
            bbox = obj['bbox'] # xyhw
            if area > image_area * self.area_ratio:
                obj_ids.append(i)
                obj_areas.append(area)
                obj_bbox.append(bbox)

        if len(obj_bbox) < 2:
            print(f"[Info] Skip image index {image_name[:-4]} due to insufficient bbox.")
            return

        # filter by IOU
        bbox_xyxy = []
        for box in obj_bbox:
            x, y, w, h = box
            bbox_xyxy.append([x, y, x + w, y + h])
        bbox_xyxy = np.array(bbox_xyxy)  # shape: [N, 4]

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

        os.makedirs(Path(self.construct_dataset_dir) / image_name[:-4], exist_ok=True)
        # cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name[:-4] / "image.jpg"), image) # source image
        dst = Path(self.construct_dataset_dir) / image_name[:-4] / "image.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, dst)

        box0 = obj_bbox[index0]
        box1 = obj_bbox[index1]

        anno_id = anno[obj_ids[index0]]
        mask = lvis_api.ann_to_mask(anno_id)
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

        anno_id = anno[obj_ids[index1]]
        mask = lvis_api.ann_to_mask(anno_id)
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
        sample_list = os.listdir(self.construct_dataset_dir)
        sample_path = os.path.join(self.construct_dataset_dir, sample_list[idx])

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
    two-object case: train/test: 44935/8859
    '''
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Objects365Dataset Analysis")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--construct_dataset_dir", type=str, default='bin', help="Path to the debug bin directory.")
    parser.add_argument("--dataset_name", type=str, default='object365', help="Dataset name.")
    parser.add_argument('--is_train', action='store_true', help="Train/Test")
    parser.add_argument('--is_build_data', action='store_true', help="Build data")
    parser.add_argument('--is_multiple', action='store_true', help="Multiple/Two objects")
    parser.add_argument("--area_ratio", type=float, default=0.01171, help="Area ratio for filtering out small objects.")
    parser.add_argument("--obj_thr", type=int, default=20, help="Object threshold for filtering.")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to test.")
    args = parser.parse_args()

    if args.is_train: 
        image_dir = Path(args.dataset_dir) / args.dataset_name / "images" / "train"
        json_dir = Path(args.dataset_dir) / args.dataset_name / "labels" / "train"
        max_num = 20000000
    else:
        image_dir = Path(args.dataset_dir) / args.dataset_name / "images" / "val"
        json_dir = Path(args.dataset_dir) / args.dataset_name / "labels" / "val"
        max_num = 3000000

    dataset = Objects365Dataset(
        json_dir = json_dir, 
        construct_dataset_dir = args.construct_dataset_dir,
        obj_thr = args.obj_thr, 
        area_ratio = args.area_ratio, 
    )

    if args.is_build_data: 
        if not args.is_multiple:
            for index in range(max_num):
                dataset._intersect_2_obj(image_dir, json_dir, index)
    else:
        for index in range(len(os.listdir(args.construct_dataset_dir))):
            collage = dataset._get_sample(index)
