import cv2
import numpy as np
import os
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
from pathlib import Path
from util.box_ops import mask_to_bbox_xywh, draw_bboxes, compute_iou_matrix
import shutil

IS_VERIFY = False

class VITONHDDataset(BaseDataset):
    def __init__(self, construct_dataset_dir, obj_thr=20, area_ratio=0.02):
        self.obj_thr = obj_thr
        self.construct_dataset_dir = construct_dataset_dir
        os.makedirs(Path(self.construct_dataset_dir), exist_ok=True)
        self.area_ratio = area_ratio
        self.sample_list = os.listdir(self.construct_dataset_dir)

    def _intersect_2_obj(self, asset_dir, idx):
        image_dir = os.path.join(asset_dir, 'image')
        image_list = os.listdir(image_dir)
        image_path = os.path.join(image_dir, image_list[idx])
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)

        mask_dir = os.path.join(asset_dir, 'image-parse-v3')
        segmentation_path = os.path.join(mask_dir, image_name[:-4]+'.png')
        segmentation = Image.open(segmentation_path).convert('P')
        segmentation = np.array(segmentation)

        h, w = image.shape[0:2]
        image_area = h*w

        ids = np.unique(segmentation)
        ids = [ i for i in ids if i!=0 ] # remove background mask
        if len(ids) < 2:
            print(f"[Info] Skip image index {image_name[:-4]} due to insufficient bbox.")
            return

        # filter by area
        obj_ids = []
        obj_areas = []
        obj_bbox = []
        for i in ids:
            mask_id = (segmentation == int(i)).astype(np.uint8)
            bbox = mask_to_bbox_xywh(mask_id) # xyhw
            area = np.sum(mask_id)
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
            os.makedirs(Path(self.construct_dataset_dir) / image_name[:-4], exist_ok=True)
            image_with_boxes = draw_bboxes(image, bbox_xyxy)
            cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name[:-4] / "bboxes_image.png"), image_with_boxes)
        
        iou_matrix = compute_iou_matrix(bbox_xyxy)
        np.fill_diagonal(iou_matrix, -1) # Exclude self-comparisons (i.e., each box with itself)

        # max_index = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        # index0, index1 = max_index[0], max_index[1]
        # max_iou = iou_matrix[index0, index1]

        # 按面积从大到小排序
        sorted_obj_ids = np.argsort(obj_areas)[::-1]
        assert len(sorted_obj_ids) > 0

        # 用面积最大的对象的索引作为 index0
        index0 = sorted_obj_ids[0]

        # 在 iou_matrix 中找到该对象对应的最佳匹配 index1
        index1 = sorted_obj_ids[1]

        # 得到对应的最大 iou
        # max_iou = iou_matrix[index0, index1]

        # if max_iou <= 0:
        #     print(f"[Info] Skip image index {image_name[:-4]} due to no overlapping bboxes.")
        #     return

        os.makedirs(Path(self.construct_dataset_dir) / image_name[:-4], exist_ok=True)
        dst = Path(self.construct_dataset_dir) / image_name[:-4] / "image.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, dst)

        mask = (segmentation == int(obj_ids[index0])).astype(np.uint8)
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

        mask = (segmentation == int(obj_ids[index1])).astype(np.uint8)
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
    two-object case: train/test: 11626/2028
    '''
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="VITONHDDataset Analysis")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--construct_dataset_dir", type=str, default='bin', help="Path to the debug bin directory.")
    parser.add_argument("--dataset_name", type=str, default='VitonHD', help="Dataset name.")
    parser.add_argument('--is_train', action='store_true', help="Train/Test")
    parser.add_argument('--is_build_data', action='store_true', help="Build data")
    parser.add_argument('--is_multiple', action='store_true', help="Multiple/Two objects")
    parser.add_argument("--area_ratio", type=float, default=0.01171, help="Area ratio for filtering out small objects.")
    parser.add_argument("--obj_thr", type=int, default=20, help="Object threshold for filtering.")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to test.")
    args = parser.parse_args()

    if args.is_train:
        asset_dir = Path(args.dataset_dir) / args.dataset_name / "train"
    else:
        asset_dir = Path(args.dataset_dir) / args.dataset_name / "test"

    dataset = VITONHDDataset(
        construct_dataset_dir = args.construct_dataset_dir,
        obj_thr = args.obj_thr, 
        area_ratio = args.area_ratio, 
    )

    max_num = 20000

    if args.is_build_data: 
        if not args.is_multiple:
            for index in range(max_num):
                dataset._intersect_2_obj(asset_dir, index)
    else:
        for index in range(len(os.listdir(args.construct_dataset_dir))):
            collage = dataset._get_sample(index)
