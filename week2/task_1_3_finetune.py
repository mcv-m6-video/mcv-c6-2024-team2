# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
import pandas as pd
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from cached_property import cached_property
from functools import cached_property
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import matplotlib.pyplot as plt

from utils import parse_xml_annotations

path_output = 'outputs/'
path_input = 'frames/'
path_xml = 'ai_challenge_s03_c010-full_annotation.xml'
model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x'
threshold = 0.5

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

    dataset_dicts = []
    for i, img in enumerate(images):
        img_item = {}
        img_item['file_name'] = os.path.join(img_dir, img)
        img_item['height'] = height
        img_item['width'] = width
        img_item['image_id'] = i

        objs = []
        if i in gt_boxes:
            for b in gt_boxes[i]:
                x = int(b['xtl'])
                y = int(b['ytl'])
                x_2 = int(b['xbr'])
                y_2 = int(b['ybr'])

                bbox = [x, y, x_2, y_2]
                bbox = list(map(float, bbox))

                class_id = 1 if b['label'] == 'bicycle' else 0

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_id,
                }
                objs.append(obj)

        img_item['annotations'] = objs
        dataset_dicts.append(img_item)

    return dataset_dicts

DatasetCatalog.register("seq_03_finetune", get_frame_dicts)
MetadataCatalog.get("seq_03_finetune").set(thing_classes=["car", "bicycle"])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(f"{model}.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

cfg.DATASETS.TRAIN = ("seq_03_finetune",)
cfg.DATASETS.TEST = ()  
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{model}.yaml")

# Fine-tuning
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []  
cfg.SOLVER.WARMUP_ITERS = 100

cfg.OUTPUT_DIR = "fine_tuned_output/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Initialize the model with pre-trained weights
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

# Fine-tune the model
trainer.train()

# Set up the Evaluator
evaluator = COCOEvaluator("seq_03_finetune", output_dir="fine_tuned_output_evaluation")
val_loader = build_detection_test_loader(cfg, "seq_03_finetune")

# Load the fine-tuned model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
predictor = DefaultPredictor(cfg)

# Run inference and evaluate
result = inference_on_dataset(predictor.model, val_loader, evaluator)
print(result)


output_file_path = "output_cars_predictions_finetuned.json"
predictions = {}
for img in os.listdir(path_input):
    im_path = os.path.join(path_input, img)
    im = cv2.imread(im_path)

    outputs = predictor(im)
    predictions[img] = []

    instances = outputs["instances"]
    car_detections = instances[instances.pred_classes == 2]

    for box in car_detections.pred_boxes.to('cpu'):
        predictions[img].append(box.tolist())

with open(output_file_path, "w") as output_file:
    json.dump(predictions, output_file, indent=2)

train_losses = []
val_losses = []
val_accuracies = []

class FineTuningMetricsHook:
    def __init__(self, trainer):
        self.trainer = trainer
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def after_step(self):
        metrics = self.trainer.storage.latest_metrics
        if metrics is not None:
            # Track training loss
            train_loss = metrics.get("total_loss")
            if train_loss is not None:
                self.train_losses.append(train_loss)

            # Track validation loss and accuracy
            val_loss = metrics.get("validation/total_loss")
            val_accuracy = metrics.get("validation/AP")
            if val_loss is not None and val_accuracy is not None:
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

fine_tuning_metrics_hook = FineTuningMetricsHook(trainer)
evaluator = COCOEvaluator("seq_03_finetune", output_dir="fine_tuned_output_evaluation")
val_loader = build_detection_test_loader(cfg, "seq_03_finetune")

trainer = DefaultTrainer(cfg)
trainer.register_hooks([fine_tuning_metrics_hook])
trainer.resume_or_load(resume=False)

# Run fine-tuning
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
predictor = DefaultPredictor(cfg)

result = inference_on_dataset(predictor.model, val_loader, evaluator)
print(result)

output_evaluation_file_path = "/ghome/group04/MCV-C5-G4/week2/c6-diana/fine_tuned_output/evaluation_results.txt"
with open(output_evaluation_file_path, "w") as output_file:
    output_file.write(str(result))

plt.plot(fine_tuning_metrics_hook.train_losses, label="Training Loss")
plt.plot(fine_tuning_metrics_hook.val_losses, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs. Validation Loss during Fine-tuning")

output_plot_file_path = "/ghome/group04/MCV-C5-G4/week2/c6-diana/fine_tuned_output/training_vs_validation_loss_plot.png"
plt.savefig(output_plot_file_path)

hyperparameter_table = pd.DataFrame({
    "Learning Rate": [cfg.SOLVER.BASE_LR],
    "Epochs": [cfg.SOLVER.MAX_ITER],
    "Batch Size": [cfg.SOLVER.IMS_PER_BATCH],
    "Image Size": [cfg.INPUT.MAX_SIZE_TEST],
    "Momentum": [cfg.SOLVER.MOMENTUM],
    "Weight Decay": [cfg.SOLVER.WEIGHT_DECAY],
    "Warm-up Epochs": [cfg.SOLVER.WARMUP_ITERS],
})

print("Hyperparameter Configuration:")
print(hyperparameter_table)

output_visualization_path = "/ghome/group04/MCV-C5-G4/week2/c6-diana/fine_tuned_output/visualizations/"
os.makedirs(output_visualization_path, exist_ok=True)

for img in os.listdir(path_input):
    im_path = os.path.join(path_input, img)
    im = cv2.imread(im_path)

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    output_visualization_file_path = os.path.join(output_visualization_path, img)
    cv2.imwrite(output_visualization_file_path, out.get_image()[:, :, ::-1])

print(f"Visualization images saved in: {output_visualization_path}")
