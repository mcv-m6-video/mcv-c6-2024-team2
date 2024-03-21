import os
import cv2
from .convertor import get_bboxes

def get_frame_number(path):
    with open(path, 'r') as file:
        data = file.read()

    # Splitting the data into lines
    lines = data.split('\n')[:-1] # Until -1 to remove the last empty line

    frames = []
    for line in lines:
        parts = line.split(',')
        frames.append(int(parts[0]))

    return min(frames), max(frames)

def load_frame_number(dir_path, seq, det_name='det/det_yolo3.txt'):
    camdirs = sorted(os.listdir(os.path.join(dir_path, seq)))
    frame_boundaries = []

    for camdir in camdirs:
        path = os.path.join(dir_path, seq, camdir, det_name)
        frame_min, frame_max = get_frame_number(path)

        frame_boundaries.append([frame_min, frame_max])
    
    return frame_boundaries

def load_sequence(dir_path, img_path, seq, video_name='vdo.avi', det_name='det/det_yolo3.txt', with_id=False):
    video_captures, bboxes = [], []
    camdirs = sorted(os.listdir(os.path.join(dir_path, seq)))
    for camdir in camdirs:
        path = os.path.join(dir_path, seq, camdir, video_name)
        video_captures.append(cv2.VideoCapture(path))

        for img_name in os.listdir(img_path):
            if seq[-1] in img_name: # Needs to be done like this since seq03 and se04 are in the same picture
                img = cv2.imread(os.path.join(img_path, img_name))

        path = os.path.join(dir_path, seq, camdir, det_name)
        bboxex_per_camera = get_bboxes(path, with_id)
        bboxes.append(bboxex_per_camera)
    
    return video_captures, img, bboxes