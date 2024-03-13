import cv2
import numpy as np
from utils import parse_xml_annotations
from PIL import Image, ImageDraw
import os
import json

dir_path = 'AICity_data/train/S03/c010'
video_path = dir_path + '/vdo.avi'
roi_path = dir_path + '/roi.jpg'
xml_path = dir_path + '/ai_challenge_s03_c010-full_annotation.xml'
frames_path = 'frames_roi/'
json_file_path = 'task1_1_predictions_all.json'

img = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(video_path)

width = cap.get(3) 
height = cap.get(4) 

gt_boxes = parse_xml_annotations(xml_file=xml_path)
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width_intersection = max(0, x2 - x1)
    height_intersection = max(0, y2 - y1)

    area_intersection = width_intersection * height_intersection

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / area_union if area_union > 0 else 0.0
    return iou

def images_to_gif(frames, output_path):
    gif_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    gif_frames = [Image.fromarray(frame) for frame in gif_frames]
    gif_frames[0].save(output_path, 
            save_all = True, append_images = gif_frames[1:], 
            optimize = True, duration = 10, loop=0, quality=80) 

def track_video(cap, start_frame=0, end_frame=None, mode='show', roi=False):
    current_frames = {}
    previous_frames = {}
    tracking_id = 1

    threshold = 0.2

    gif_frames = []

    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 1
    color = (255, 0, 0) 
    thickness = 2

    frame_id = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Play the video
    while cap.isOpened(): 

        # Capture frame-by-frame 
        ret, frame = cap.read() 

        if not ret or (end_frame is not None and frame_id >= end_frame):
            break

        previous_frames = current_frames
        current_frames = {}

        if roi:
            roi_coord = np.where(img == 0)
            frame[roi_coord[0], roi_coord[1]] = 0
            
        curr_frame = 'frame' + str(frame_id) + '.png'
        if frame_id == 0:
            for b in json_data[curr_frame]:
                b = list(map(int, b))

                current_frames[tracking_id] = b
                tracking_id += 1

        else:
            if curr_frame in json_data:
                for b in json_data[curr_frame]:
                    highest_overlap = {}
                    b = list(map(int, b))

                    for k, prev_b in previous_frames.items():
                        iou = calculate_iou(b, prev_b)
                        if iou > threshold:
                            highest_overlap[k] = iou
                        
                    if len(highest_overlap.keys()) > 0:
                        key_max = max(highest_overlap, key=highest_overlap.get)
                        current_frames[key_max] = b

                    else:
                        current_frames[tracking_id] = b
                        tracking_id += 1


        for k, b in current_frames.items():
            x, y, x_2, y_2 = b
            cv2.putText(frame, str(k), (x, y), font, fontScale, color, thickness)
            cv2.rectangle(frame, (x, y), (x_2, y_2), (0, 255, 0), 2)


        if mode == 'show':
            cv2.imshow('Foreground detection', frame)
        elif mode == 'write':
            cv2.imwrite(f'{frames_path}frame{frame_id}.png', frame)

        gif_frames.append(frame)

        frame_id += 1

        # Press Q on keyboard to exit 
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    images_to_gif(gif_frames, 'tracked_gt_2.gif')


track_video(cap, start_frame=320, end_frame=365)