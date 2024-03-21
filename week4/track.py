import cv2
import numpy as np
import os
import json
from utils import track_to_file, get_bboxes
from sort.sort import Sort
import argparse

det_name_in = 'det/det_yolo3.txt'
dir_path = './AICity/train'

def get_tracked_bboxes(bboxes):
    output_bboxes = {}

    mot_tracker = Sort() 
    for frame_id in list(bboxes.keys()):
        frame_bboxes = bboxes[frame_id]
        frame_bboxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in frame_bboxes]

        track_bboxes = mot_tracker.update(np.array(frame_bboxes))
        output_bboxes[frame_id] = track_bboxes

    return output_bboxes


def track_sequence(dir_path, seq, det_name_in):
    det_name_out = 'det/det_tracked.txt'

    seq = os.path.join(dir_path, seq)
    for camdir in os.listdir(seq):
        path = os.path.join(seq, camdir, det_name_in)
        bboxex_per_camera = get_bboxes(path)

        tracked = get_tracked_bboxes(bboxex_per_camera)
        track_to_file(tracked, os.path.join(seq, camdir, det_name_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SORT tracking and save files for given sequence.")
    parser.add_argument("seq", choices=["S01", "S03", "S04"], help="Sequence option (S01, S03, S04)")
    args = parser.parse_args()

    track_sequence(dir_path, args.seq, det_name_in)
