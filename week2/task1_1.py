# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import time
import shutil
from tqdm import tqdm

from utils import parse_xml_annotations, save_frames

# import some common detectron2 utilities
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


path_output = './outputs/'
path_input = './frames/'
path_xml = './ai_challenge_s03_c010-full_annotation.xml'
model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x'
threshold = 0.5

dir_path = 'AICity_data/train/S03/c010/'
video_path = dir_path + 'vdo.avi'
frames_path = 'frames/'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(f"{model}.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{model}.yaml")
predictor = DefaultPredictor(cfg)

if not os.path.exists(path_input):
    os.makedirs(path_input)
    cap = cv2.VideoCapture(video_path)
    save_frames(cap, path_input, mode='write')


def get_images(img_dir):
    images = [img for img in os.listdir(img_dir)]

    return sorted(images, key=lambda x: x[0])

def get_frame_dicts():
    img_dir = path_input
    xml_path = path_xml
    height = 1080
    width = 1920

    gt_boxes = parse_xml_annotations(xml_file=xml_path)

    images = get_images(img_dir)
    print(images)

    dataset_dicts = []
    for i, img in enumerate(images):

        img_item = {}
        img_item['file_name'] = img_dir + img
        img_item['height']= height
        img_item['width']= width
        img_item['image_id']= i

        objs = []
        if i in gt_boxes:
            for b in gt_boxes[i]:
                x = int(b['xtl'])
                y = int(b['ytl'])
                x_2 = int(b['xbr'])
                y_2 = int(b['ybr'])

                bbox = [x, y, x_2, y_2]
                bbox = list(map(float, bbox))

                if b['label'] == 'bycicle':
                    class_id = 1
                elif b['label'] == 'car':
                    class_id = 0    

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_id,
                }
                objs.append(obj)
        
        img_item['annotations'] = objs
        dataset_dicts.append(img_item)
    
    return dataset_dicts

DatasetCatalog.register("seq_03", get_frame_dicts)
MetadataCatalog.get("seq_03").set(thing_classes=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)

evaluator = COCOEvaluator("seq_03", output_dir="evaluate_results")
val_loader = build_detection_test_loader(cfg, "seq_03")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
