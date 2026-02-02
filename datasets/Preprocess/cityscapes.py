import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
from scipy.ndimage import label
import json
from abc import ABCMeta, abstractmethod
import PIL.ImageDraw as ImageDraw
from util.box_ops import compute_iou_matrix, draw_bboxes
from util.cityscapes_ops import Annotation, name2label

IS_VERIFY = True

class CityscapesDataset(BaseDataset):
    def __init__(self, construct_dataset_dir, obj_thr=20, area_ratio=0.02):
        self.obj_thr = obj_thr
        self.dynamic = 0
        self.construct_dataset_dir = construct_dataset_dir
        self.area_ratio = area_ratio

        # self.image_dir = image_dir
        # self.label_path = label_path
        # self.obj_thr = obj_thr

        # self.file_list = sorted(os.listdir(label_path))
        # self.dynamic = 0
        # self.data = os.listdir(image_dir)


    def _intersect_2_obj(self, image_dir, json_dir, idx):
        json_list = os.listdir(label_path)
        image_name = json_list[idx][:-21]
        image_path = os.path.join(self.image_dir, image_name+'_leftImg8bit.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        json_path = os.path.join(json_dir, image_name+'_gtFine_polygons.json')
        annotation = Annotation()
        annotation.fromJsonFile(json_path)
        size = (annotation.imgWidth, annotation.imgHeight)

        encoding="color"
        outline=None

        # the background
        if encoding == "ids":
            background = name2label['unlabeled'].id
        elif encoding == "trainIds":
            background = name2label['unlabeled'].trainId
        elif encoding == "color":
            background = name2label['unlabeled'].color
        else:
            # print("Unknown encoding '{}'".format(encoding))
            return None


    def get_sample(self, idx):
        image_name = self.file_list[idx][:-21]
        image_path = os.path.join(self.image_dir, image_name+'_leftImg8bit.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_image_path = os.path.join(self.label_path, image_name+'_gtFine_polygons.json')
        annotation = Annotation()
        annotation.fromJsonFile(label_image_path)
        size = ( annotation.imgWidth , annotation.imgHeight )

        encoding="color"
        outline=None

        # the background
        if encoding == "ids":
            background = name2label['unlabeled'].id
        elif encoding == "trainIds":
            background = name2label['unlabeled'].trainId
        elif encoding == "color":
            background = name2label['unlabeled'].color
        else:
            # print("Unknown encoding '{}'".format(encoding))
            return None


        obj_ids = []
        obj_areas = []
        counter = 0
        # loop over all objects
        for obj in annotation.objects:
            label   = obj.label
            polygon = obj.polygon

            # If the object is deleted, skip it
            if obj.deleted:
                continue

            # If the label is not known, but ends with a 'group' (e.g. cargroup)
            # try to remove the s and see if that works
            if ( not label in name2label ) and label.endswith('group'):
                label = label[:-len('group')]

            if not label in name2label:
                printError( "Label '{}' not known.".format(label) )

            # If the ID is negative that polygon should not be drawn
            if name2label[label].id < 0:
                continue

            # print((name2label[label].id).type)

            if name2label[label].id !=26 and name2label[label].id !=27 and name2label[label].id !=28:
                continue

            if encoding == "ids":
                val = name2label[label].id
            elif encoding == "trainIds":
                val = name2label[label].trainId
            elif encoding == "color":
                val = name2label[label].color

            # print(val)

            if encoding == "color":
                labelImg = Image.new("RGBA", size, background)
            else:
                labelImg = Image.new("L", size, background)

            drawer = ImageDraw.Draw( labelImg )

            try:
                if outline:
                    drawer.polygon( polygon, fill=(255, 255, 255), outline=outline )
                else:
                    drawer.polygon( polygon, fill=(255, 255, 255))
            except:
                # print("Failed to draw polygon with label {}".format(label))
                raise

            area = np.sum(labelImg)

            if area > 3600:
                obj_ids.append(counter)
                obj_areas.append(area)
            counter+=1

        assert len(obj_ids) > 0

        sorted_obj_ids = np.argsort(obj_areas)[::-1]
        assert len(sorted_obj_ids) > 0
        if len(sorted_obj_ids) >= self.obj_thr:
            sorted_obj_ids = sorted_obj_ids[:self.obj_thr]

        assert len(sorted_obj_ids) >= self.obj_thr

        item_with_collage = {}
        counter = 0
        counter_ = 0
        # loop over all objects
        for obj in annotation.objects:
            label   = obj.label
            polygon = obj.polygon

            # If the object is deleted, skip it
            if obj.deleted:
                continue

            # If the label is not known, but ends with a 'group' (e.g. cargroup)
            # try to remove the s and see if that works
            if ( not label in name2label ) and label.endswith('group'):
                label = label[:-len('group')]

            if not label in name2label:
                printError( "Label '{}' not known.".format(label) )

            # If the ID is negative that polygon should not be drawn
            if name2label[label].id < 0:
                continue

            # print((name2label[label].id).type)

            if name2label[label].id !=26 and name2label[label].id !=27 and name2label[label].id !=28:
                continue

            if encoding == "ids":
                val = name2label[label].id
            elif encoding == "trainIds":
                val = name2label[label].trainId
            elif encoding == "color":
                val = name2label[label].color

            if encoding == "color":
                labelImg = Image.new("RGBA", size, background)
            else:
                labelImg = Image.new("L", size, background)

            drawer = ImageDraw.Draw( labelImg )

            try:
                if outline:
                    drawer.polygon( polygon, fill=(255, 255, 255), outline=outline )
                else:
                    drawer.polygon( polygon, fill=(255, 255, 255))
            except:
                # print("Failed to draw polygon with label {}".format(label))
                raise

            if counter in sorted_obj_ids:
                # print(np.shape(image), np.shape(labelImg))
                # print(image.type, labelImg.type)
                # patch = self.get_patch(image, np.asarray(labelImg)[:, :, 0]/255)
                ref_mask = np.asarray(labelImg)[:, :, 0]/255
                patch_dir = os.path.join(os.path.dirname(self.image_dir), 'patch_train', 'original', f"{image_name}_{counter}")
                item_id = self.process_pairs_multiple(image, ref_mask, patch_dir, counter_)
                item_with_collage.update(item_id)
                counter_+=1

            counter+=1

        # composition
        item_with_collage = self.process_composition(item_with_collage, self.obj_thr)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['object_num'] = self.obj_thr
        return item_with_collage#, image_name, self.obj_thr

    def get_patches(self, idx, save_dir):
        image_name = self.file_list[idx][:-21]
        image_path = os.path.join(self.image_dir, image_name+'_leftImg8bit.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image_name) # aachen_000000_000019
        # print(np.shape(ref_image)) # (1024, 2048, 3) 

        label_image_path = os.path.join(self.label_path, image_name+'_gtFine_polygons.json')
        # with open(label_image_path) as f:
            # ann_info = json.load(f)
            # print(ann_info)

        annotation = Annotation()
        annotation.fromJsonFile(label_image_path)

        # the size of the image
        size = ( annotation.imgWidth , annotation.imgHeight )

        encoding="color"
        outline=None

        # the background
        if encoding == "ids":
            background = name2label['unlabeled'].id
        elif encoding == "trainIds":
            background = name2label['unlabeled'].trainId
        elif encoding == "color":
            background = name2label['unlabeled'].color
        else:
            # print("Unknown encoding '{}'".format(encoding))
            return None

        # # this is the image that we want to create
        # if encoding == "color":
        #     labelImg = Image.new("RGBA", size, background)
        # else:
        #     labelImg = Image.new("L", size, background)

        # a drawer to draw into the image
        # drawer = ImageDraw.Draw( labelImg )

        obj_ids = []
        obj_areas = []
        counter = 0
        # loop over all objects
        for obj in annotation.objects:
            label   = obj.label
            polygon = obj.polygon

            # If the object is deleted, skip it
            if obj.deleted:
                continue

            # If the label is not known, but ends with a 'group' (e.g. cargroup)
            # try to remove the s and see if that works
            if ( not label in name2label ) and label.endswith('group'):
                label = label[:-len('group')]

            if not label in name2label:
                printError( "Label '{}' not known.".format(label) )

            # If the ID is negative that polygon should not be drawn
            if name2label[label].id < 0:
                continue

            # print((name2label[label].id).type)

            if name2label[label].id !=26 and name2label[label].id !=27 and name2label[label].id !=28:
                continue

            if encoding == "ids":
                val = name2label[label].id
            elif encoding == "trainIds":
                val = name2label[label].trainId
            elif encoding == "color":
                val = name2label[label].color

            # print(val)

            if encoding == "color":
                labelImg = Image.new("RGBA", size, background)
            else:
                labelImg = Image.new("L", size, background)

            drawer = ImageDraw.Draw( labelImg )

            # try:
            #     if outline:
            #         drawer.polygon( polygon, fill=val, outline=outline )
            #     else:
            #         drawer.polygon( polygon, fill=val)
            # except:
            #     print("Failed to draw polygon with label {}".format(label))
            #     raise

            try:
                if outline:
                    drawer.polygon( polygon, fill=(255, 255, 255), outline=outline )
                else:
                    drawer.polygon( polygon, fill=(255, 255, 255))
            except:
                # print("Failed to draw polygon with label {}".format(label))
                raise


            # labelImg.save( 'tmp/'+str(counter)+'.png' )
            area = np.sum(labelImg)

            if area > 3600:
                obj_ids.append(counter)
                obj_areas.append(area)
            counter+=1

        assert len(obj_ids) > 0

        sorted_obj_ids = np.argsort(obj_areas)[::-1]
        assert len(sorted_obj_ids) > 0
        if len(sorted_obj_ids) >= self.obj_thr:
            sorted_obj_ids = sorted_obj_ids[:self.obj_thr]
        # else: 
            # sorted_obj_ids = np.concatenate((sorted_obj_ids, sorted_obj_ids))

        # print(len(sorted_obj_ids))
        assert len(sorted_obj_ids) >= self.obj_thr

        counter = 0
        # loop over all objects
        for obj in annotation.objects:
            label   = obj.label
            polygon = obj.polygon

            # If the object is deleted, skip it
            if obj.deleted:
                continue

            # If the label is not known, but ends with a 'group' (e.g. cargroup)
            # try to remove the s and see if that works
            if ( not label in name2label ) and label.endswith('group'):
                label = label[:-len('group')]

            if not label in name2label:
                printError( "Label '{}' not known.".format(label) )

            # If the ID is negative that polygon should not be drawn
            if name2label[label].id < 0:
                continue

            if name2label[label].id !=26 and name2label[label].id !=27 and name2label[label].id !=28:
                continue

            if encoding == "ids":
                val = name2label[label].id
            elif encoding == "trainIds":
                val = name2label[label].trainId
            elif encoding == "color":
                val = name2label[label].color

            if encoding == "color":
                labelImg = Image.new("RGBA", size, background)
            else:
                labelImg = Image.new("L", size, background)

            drawer = ImageDraw.Draw( labelImg )

            try:
                if outline:
                    drawer.polygon( polygon, fill=(255, 255, 255), outline=outline )
                else:
                    drawer.polygon( polygon, fill=(255, 255, 255))
            except:
                # print("Failed to draw polygon with label {}".format(label))
                raise

            if counter in sorted_obj_ids:
                patch = self.get_patch(image, np.asarray(labelImg)[:, :, 0]/255)
                image_path = os.path.join(save_dir, f"{image_name}_{counter}.jpg")
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_path, patch)  # RGB -> BGR


            counter+=1


        return



    def __len__(self):
        return 2975

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


if __name__ == "__main__":
    import argparse
    import pprint

    image_dir = "/data/hang/customization/data/Cityscapes/image_train"
    label_path = "/data/hang/customization/data/Cityscapes/json_train"
    obj_thr = 2
    index = 33

    # 初始化数据集
    dataset = CityscapesDataset(
        image_dir=image_dir,
        label_path=label_path,
        obj_thr=obj_thr
    )


if __name__ == "__main__":
    '''
    two-object case: train/test: 44935/8859
    '''
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="CityscapesDataset Analysis")
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

    if args.is_train:
        image_dir = Path(args.dataset_dir) / args.dataset_name / "train" / "images"
        json_dir = Path(args.dataset_dir) / args.dataset_name / "train" / "jsons"
        max_num = 2975
    else:
        image_dir = Path(args.dataset_dir) / args.dataset_name / "val" / "images"
        json_dir = Path(args.dataset_dir) / args.dataset_name / "val" / "jsons"
        max_num = 500

    dataset = CityscapesDataset(
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
                dataset._intersect_2_obj(image_dir, json_dir, index)
    else:
        for index in range(len(os.listdir(args.construct_dataset_dir))):
            collage = dataset._get_sample(index)


            
