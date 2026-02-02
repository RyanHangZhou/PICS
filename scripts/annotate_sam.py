import json
import torch
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor
from lvis import LVIS
import copy
from pathlib import Path
# import time


class Objects365SAM():
    def __init__(self, index_low, index_high):
        # Load SAM model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

        self.index_low = index_low
        self.index_high = index_high

    # Load annotations
    def load_annotations(self, annotation_file):
        with open(annotation_file, 'r') as f:
            self.json_data = json.load(f)

    def process_annotations_with_sam(self, images_dir, output_dir):
        image_info_list = self.json_data['images']
        counter = 0
        for image_info in image_info_list[self.index_low:self.index_high]:
            # start_time = time.time()
            image_id = image_info['id']
            image_name = image_info['file_name'].split('/')[-1]
            image_subset = image_info['file_name'].split('/')[-2]

            output_json_dir = Path(os.path.join(output_dir, image_subset))
            output_json_dir.mkdir(exist_ok=True)

            image_path = os.path.join(images_dir, image_subset, image_name)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image not found: {image_path}")
                continue
            h, w, _ = image.shape
            self.predictor.set_image(image)

            # Get annotations for this image
            image_annotations = [anno for anno in self.json_data['annotations'] if anno['image_id'] == image_id]

            # Create bounding boxes from COCO format
            bounding_boxes = []
            for anno in image_annotations:
                xmin, ymin, width, height = anno['bbox']
                xmax, ymax = xmin + width, ymin + height
                bounding_boxes.append([xmin, ymin, xmax, ymax])

            # Convert bounding boxes to tensor for SAM
            input_boxes = torch.tensor(bounding_boxes, device=self.device).float()
            transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

            # Get masks from SAM
            with torch.no_grad():
                masks, scores, logits = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

            # Convert masks to COCO-style annotations
            mask_annotations = []
            for mask in masks:
                binary_mask = mask.squeeze().cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue
                largest_contour = max(contours, key=cv2.contourArea)
                segmentation = largest_contour.flatten().tolist()
                x, y, w, h = cv2.boundingRect(largest_contour)
                area = float(cv2.contourArea(largest_contour))
                # mask_annotations.append(segmentation)
                mask_annotations.append({
                    "segmentation": [segmentation],
                    "bbox": [x, y, w, h],
                    "area": area,
                    "category_id": 1
                })

            save_annotations_to_json(image_id, 
                mask_annotations, 
                os.path.join(output_json_dir, image_name[:-4]+'.json')
                )
            torch.cuda.empty_cache()
            counter += 1
            print('Done image idex: ', counter)
            # end_time = time.time()
            # total_time = end_time - start_time
            # print(f"[Timer] Avg per image: {total_time:.2f} sec, or {total_time/60:.2f} min.")

def save_annotations_to_json(image_id, mask_annotations, output_file):
    coco_format_output = {
        "image_id": image_id,
        "annotations": mask_annotations
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format_output, f)


if __name__ == "__main__":
    '''
    Image number: train/test: 1742292/80000
    '''
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Annotate labels with Segment Anything")
    parser.add_argument('--is_train', action='store_true', help="Train/Test")
    parser.add_argument("--index_low", type=int, default=0, help="Lower bound of indexes for processing Objects365 dataset.")
    parser.add_argument("--index_high", type=int, default=1742292, help="Upper bound of indexes for processing Objects365 dataset.")
    args = parser.parse_args()

    if args.is_train: 
        input_json_dir = '../data/object365/zhiyuan_objv2_train.json'
        input_image_dir = '../data/object365/images/train/'
        output_dir = Path('../data/object365/labels/train/')
    else:
        input_json_dir = '../data/object365/zhiyuan_objv2_val.json'
        input_image_dir = '../data/object365/images/val/'
        output_dir = Path('../data/object365/labels/val/')

    output_dir.mkdir(exist_ok=True)

    sam_annotator = Objects365SAM(args.index_low, args.index_high)
    sam_annotator.load_annotations(input_json_dir)
    sam_annotator.process_annotations_with_sam(input_image_dir, output_dir)


