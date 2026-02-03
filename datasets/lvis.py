import cv2
import numpy as np
import os
from .data_utils import * 
from .base import BaseDataset
from lvis import LVIS
from pathlib import Path
from util.box_ops import compute_iou_matrix, draw_bboxes
import shutil

IS_VERIFY = False

class LVISDataset(BaseDataset):
    def __init__(self, construct_dataset_dir, obj_thr=20, area_ratio=0.02):
        self.obj_thr = obj_thr
        self.construct_dataset_dir = construct_dataset_dir
        os.makedirs(Path(self.construct_dataset_dir), exist_ok=True)
        self.area_ratio = area_ratio
        self.sample_list = os.listdir(self.construct_dataset_dir)

    def _get_image_path(self, file_name):
        for img_dir in self.image_dir:
            path = img_dir / file_name
            if path.exists():
                return str(path)
        raise FileNotFoundError(f"File {file_name} not found in any of the image_dir.")

    def _intersect_2_obj(self, image_dir, lvis_api, imgs_info, annos, idx):
        self.image_dir = image_dir
        image_name = imgs_info[idx]['coco_url'].split('/')[-1]
        image_path = self._get_image_path(image_name)
        image = cv2.imread(image_path)

        h, w = image.shape[0:2]
        image_area = h*w

        anno = annos[idx]

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
            os.makedirs(Path(self.construct_dataset_dir) / image_name[:-4], exist_ok=True)
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
        dst = Path(self.construct_dataset_dir) / image_name[:-4] / "image.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, dst)

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

    def _intersect_3_obj(self, image_dir, lvis_api, imgs_info, annos, idx):
        self.image_dir = image_dir
        image_name = imgs_info[idx]['coco_url'].split('/')[-1]
        image_path = self._get_image_path(image_name)
        image = cv2.imread(image_path)

        h, w = image.shape[0:2]
        image_area = h * w

        anno = annos[idx]

        # filter by area
        obj_ids = []
        obj_areas = []
        obj_bbox = []
        for i, obj in enumerate(anno):
            area = obj['area']
            bbox = obj['bbox']  # xywh
            if area > image_area * self.area_ratio:
                obj_ids.append(i)
                obj_areas.append(area)
                obj_bbox.append(bbox)

        if len(obj_bbox) < 3:
            print(f"[Info] Skip image index {image_name[:-4]} due to insufficient bbox (need >=3, got {len(obj_bbox)}).")
            return

        # calculate IOU matrix
        bbox_xyxy = []
        for box in obj_bbox:
            x, y, w_box, h_box = box
            bbox_xyxy.append([x, y, x + w_box, y + h_box])
        bbox_xyxy = np.array(bbox_xyxy)  # shape: [N, 4]

        if IS_VERIFY:
            os.makedirs(Path(self.construct_dataset_dir) / image_name[:-4], exist_ok=True)
            image_with_boxes = draw_bboxes(image, bbox_xyxy)
            cv2.imwrite(str(Path(self.construct_dataset_dir) / image_name[:-4] / "bboxes_image.png"), image_with_boxes)

        iou_matrix = compute_iou_matrix(bbox_xyxy)
        np.fill_diagonal(iou_matrix, -1)  # Exclude self-comparisons

        # find 3 overlapped objects
        positive_iou = np.where(iou_matrix > 0, iou_matrix, 0.0)
        row_sums = positive_iou.sum(axis=1)
        anchor = int(np.argmax(row_sums))

        partner_candidates = np.argsort(iou_matrix[anchor])[::-1]
        partners = [int(p) for p in partner_candidates if iou_matrix[anchor, p] > 0]

        if len(partners) < 2:
            print(f"[Info] Skip image index {image_name[:-4]} due to not enough overlapping bboxes for 3 objects.")
            return

        index0 = anchor
        index1 = partners[0]
        index2 = partners[1]

        max_iou_pair = max(iou_matrix[index0, index1], iou_matrix[index0, index2], iou_matrix[index1, index2])
        if max_iou_pair <= 0:
            print(f"[Info] Skip image index {image_name[:-4]} due to no overlapping bboxes.")
            return

        # copy original image
        out_dir = Path(self.construct_dataset_dir) / image_name[:-4]
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / "image.jpg"
        shutil.copy(image_path, dst)

        # first object
        anno_id = anno[obj_ids[index0]]
        mask0 = lvis_api.ann_to_mask(anno_id)
        cv2.imwrite(str(out_dir / "object_0_mask.png"), 255 * mask0)
        patch0 = self.get_patch(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask0)
        patch0 = cv2.cvtColor(patch0, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "object_0.png"), patch0)

        if IS_VERIFY:
            mask_color = np.stack([mask0 * 255] * 3, axis=-1).astype(np.uint8)
            highlight = np.zeros_like(image)
            highlight[:, :, 2] = 255  # red channel
            alpha = 0.5
            image_with_boxes = np.where(
                mask_color == 255,
                cv2.addWeighted(image_with_boxes, 1 - alpha, highlight, alpha, 0),
                image_with_boxes
            )

        # second object
        anno_id = anno[obj_ids[index1]]
        mask1 = lvis_api.ann_to_mask(anno_id)
        cv2.imwrite(str(out_dir / "object_1_mask.png"), 255 * mask1)
        patch1 = self.get_patch(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask1)
        patch1 = cv2.cvtColor(patch1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "object_1.png"), patch1)

        if IS_VERIFY:
            mask_color = np.stack([mask1 * 255] * 3, axis=-1).astype(np.uint8)
            highlight = np.zeros_like(image)
            highlight[:, :, 0] = 255  # blue channel
            alpha = 0.5
            image_with_boxes = np.where(
                mask_color == 255,
                cv2.addWeighted(image_with_boxes, 1 - alpha, highlight, alpha, 0),
                image_with_boxes
            )

        # third object
        anno_id = anno[obj_ids[index2]]
        mask2 = lvis_api.ann_to_mask(anno_id)
        cv2.imwrite(str(out_dir / "object_2_mask.png"), 255 * mask2)
        patch2 = self.get_patch(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask2)
        patch2 = cv2.cvtColor(patch2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "object_2.png"), patch2)

        if IS_VERIFY:
            mask_color = np.stack([mask2 * 255] * 3, axis=-1).astype(np.uint8)
            highlight = np.zeros_like(image)
            highlight[:, :, 1] = 255  # green channel
            alpha = 0.5
            image_with_boxes = np.where(
                mask_color == 255,
                cv2.addWeighted(image_with_boxes, 1 - alpha, highlight, alpha, 0),
                image_with_boxes
            )
            cv2.imwrite(str(out_dir / "highlighted_image.png"), image_with_boxes)


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
    two-object case: train/test: 34610/8859
    '''
    import argparse

    parser = argparse.ArgumentParser(description="LVISDataset Analysis")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--construct_dataset_dir", type=str, default='bin', help="Path to the debug bin directory.")
    parser.add_argument("--dataset_name", type=str, default='COCO', help="Dataset name.")
    parser.add_argument('--is_train', action='store_true', help="Train/Test")
    parser.add_argument('--is_build_data', action='store_true', help="Build data")
    parser.add_argument('--is_multiple', action='store_true', help="Multiple/Two objects")
    parser.add_argument("--area_ratio", type=float, default=0.01171, help="Area ratio for filtering out small objects.")
    parser.add_argument("--obj_thr", type=int, default=20, help="Object threshold for filtering.")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to test.")
    args = parser.parse_args()

    image_dirs = [
        Path(args.dataset_dir) / args.dataset_name / "train2017",
        Path(args.dataset_dir) / args.dataset_name / "val2017",
    ]

    if args.is_train: 
        json_path = Path(args.dataset_dir) / args.dataset_name / "lvis_v1/lvis_v1_train.json"
        max_num = 2000000
    else:
        json_path = Path(args.dataset_dir) / args.dataset_name / "lvis_v1/lvis_v1_val.json"
        max_num = 30000

    dataset = LVISDataset(
        construct_dataset_dir = args.construct_dataset_dir,
        obj_thr = args.obj_thr, 
        area_ratio = args.area_ratio, 
    )
    
    lvis_api = LVIS(json_path)
    img_ids = sorted(lvis_api.imgs.keys())
    imgs_info = lvis_api.load_imgs(img_ids)
    annos = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    if args.is_build_data: 
        if not args.is_multiple:
            for index in range(max_num):
                dataset._intersect_2_obj(image_dirs, lvis_api, imgs_info, annos, index)
                # dataset._intersect_3_obj(image_dirs, lvis_api, imgs_info, annos, index)
    else:
        for index in range(len(os.listdir(args.construct_dataset_dir))):
            collage = dataset._get_sample(index)
