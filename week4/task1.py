import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
from ultralytics.utils.plotting import Annotator
import argparse
#torch.cuda.set_device(0)

def is_bottom_between_lines(bbox, line1, line2):
    _, _, y_b, x_b = bbox
    
    y1, x1, y2, x2 = line1
    y3, x3, y4, x4 = line2
    
    min_x = min(x1, x2, x3, x4)
    max_x = max(x1, x2, x3, x4)
    min_y = min(y1, y2, y3, y4)
    max_y = max(y1, y2, y3, y4)

    if (x_b >= min_x and x_b <= max_x and
        y_b >= min_y and y_b <= max_y):
        return True
    else:
        return False

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Path to the input video', required=True)
parser.add_argument('-o', '--output', help='Path to save the output video', required=True)
parser.add_argument('-ot', '--outputtxt', help='Path to save the speed', required=True)
parser.add_argument('-m', '--mask', help='Path to the mask (region of interest for detection)', required=True)
parser.add_argument('-r', '--realmeasurement', type=int, help='Real measurement of the region of interest: x in meters', required=True)
parser.add_argument('-lt', '--linetop', nargs='+', type=int, help='Coordinates of the line on the top of the region of interest: y1 x, y2 x2', required=True)
parser.add_argument('-lb', '--linebottom', nargs='+', type=int, help='Coordinates of the line on the bottom of the region of interest: y1 x1 y2 x2', required=True)

args = vars(parser.parse_args())
input_path = args['input']
output_path = args['output']
output_txt_path = args['outputtxt']
real_measurement = args['realmeasurement']
line_top = args['linetop']
line_bottom = args['linebottom']
mask_path = args['mask']

cap = cv2.VideoCapture(input_path)
img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
out = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(img_width, img_height))


mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
inverted_mask = cv2.bitwise_not(mask)

model = YOLO('yolov8x.pt')

colors = {}
time = {}
speed = {}
prev_inside = {}

f = open(output_txt_path, "a")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Get foreground and background based on the mask (region of interest for detection)
    fg = cv2.bitwise_and(frame, frame, mask=mask)
    bg = cv2.bitwise_and(frame, frame, mask=inverted_mask)

    results = model.track(fg, persist=True, classes=[2, 8])
     # In case there are no detections - display and skip
    if results[0].boxes.id is None:
        cv2.line(frame, (line_top[0], line_top[1]), (line_top[2], line_top[3]), (255, 0, 0), 5)
        cv2.line(frame, (line_bottom[0], line_bottom[1]), (line_bottom[2], line_bottom[3]), (255, 0, 0), 5)
        out.write(frame)
        continue
    
    bboxes = np.array([box[:4].cpu().numpy() for box in results[0].boxes.xyxy])
    ids = np.array([id.cpu().numpy() for id in results[0].boxes.id])

    labels = []
    for bbox, id in zip(bboxes, ids):
        # Initialize color for each detected object
        if not int(id) in colors:
            color = np.random.randint(0, 255, size=(3, ))
            colors[int(id)] = (int(color[0]), int(color[1]), int(color[2]))

        is_inside = is_bottom_between_lines(bbox, line_top, line_bottom)
        if is_inside:
            if not int(id) in time: # First time the object is detected inside the region
                time[int(id)] = 1
                prev_inside[int(id)] = True
            else:
                time[int(id)] = time[int(id)] + 1
        elif is_inside == False and int(id) in prev_inside and prev_inside[int(id)] == True: # exited region => calculate speed
            prev_inside[int(id)] = False # Only calculate once
            speed[int(id)] = (real_measurement / (time[int(id)] / fps)) * 3.6
            f.write(f"Car #{int(id)}: {speed[int(id)]}\n")
            print(f"Speed for car {id} {speed[int(id)]}")

        if int(id) in speed:
            labels.append(f"#{int(id)} {int(speed[int(id)])} km/h")
        else:
            labels.append(f"#{int(id)}")

    annotator = Annotator(fg)
    boxes = results[0].boxes
    for i, (box, id) in enumerate(zip(boxes, ids)):
        b = box.xyxy[0]
        c = box.cls
        label = model.names[int(c)] + ' ' + labels[i]
        annotator.box_label(b, label, colors[int(id)])
    frame = annotator.result()

    # Add background back
    bg_mask = (frame[:, :, 1] == 0)
    frame[bg_mask] = bg[bg_mask]

    cv2.line(frame, (line_top[0], line_top[1]), (line_top[2], line_top[3]), (255, 0, 0), 5)
    cv2.line(frame, (line_bottom[0], line_bottom[1]), (line_bottom[2], line_bottom[3]), (255, 0, 0), 5)

    for bbox in bboxes:
        cv2.circle(frame, (int(bbox[2]), int(bbox[3])), 3, (255, 0, 0), 3)

    out.write(frame)
cap.release()
f.close()
out.release()