import numpy as np
import json
import os

from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm

single_cls = True       # just detect vehicle

class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None, validation_type = 'normal'):
        super().__init__(cfg, is_train, inputsize, transform)
        self.validation_type = validation_type  # Accomodates nomral validations, attack validations, and attack/defense validations
        self.db = self._get_db()
        self.cfg = cfg
        print(f"{self.validation_type}\n")


    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        # Original detection label size
        original_height, original_width = 720, 1280
        
        # Calculate scaling factor
        scale_x = width / original_width
        scale_y = height / original_height
        for mask in tqdm(list(self.mask_list)):
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
            
            # Conditional logic based on validation type
            if self.validation_type == 'normal':
                image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            else:
                base_name = os.path.splitext(os.path.basename(mask_path).split('_')[0])[0]
                image_path = os.path.join(str(self.img_root), f"{base_name}.jpg")
                
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.filter_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                if category == "traffic light":
                    color = obj['attributes']['trafficLightColor']
                    category = "tl_" + color
                if category in id_dict.keys():
                    # First scale the boundary box coordinates
                    x1 = float(obj['box2d']['x1']) * scale_x
                    y1 = float(obj['box2d']['y1']) * scale_y
                    x2 = float(obj['box2d']['x2']) * scale_x
                    y2 = float(obj['box2d']['y2']) * scale_y
                    cls_id = id_dict[category]
                    if single_cls:
                         cls_id=0
                    gt[idx][0] = cls_id
                    box = convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)
                

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass