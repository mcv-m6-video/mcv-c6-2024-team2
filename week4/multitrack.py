import cv2
import numpy as np
import os
import json
from utils import load_sequence, show_tracked, track_to_file, cossim, load_frame_number
from sort.sort import Sort
from track import track_sequence
from db import DatabaseHandler
from feature_extractor import FeatureExtractor
from PIL import Image
import argparse

dir_path = './AICity/train'
img_path = './AICity/cam_loc'
roi_path = dir_path + '/roi.jpg'
det_name = 'det/det_tracked.txt'




def recursive_track_mapping(mapping_dict, tracking_id):
    if tracking_id in mapping_dict:
        return recursive_track_mapping(mapping_dict, mapping_dict[tracking_id])
    else:
        return tracking_id



def multicamera_tracking(video_captures, bboxes, seq, dbh, fe, camdirs, frame_boundaries, det_name_out='det/det_multi.txt', start_frame=0, end_frame=None):
    """
    Parameters:
    - video_captures (list): List of videos where each video represents one camera in a sequence.
    - bboxes (list): List of dictionaries where each dictionary represents the tracked bounding boxes in one camera.
                    Each dictionary includes keys as frame indices and values as bounding box coordinates and track_id.
                    Example: [{'frame_1': [[id, x1, y1, x2, y2], [id, x1, y1, x2, y2]], 'frame_2': [[id, x1, y1, x2, y2]]}, ...]
    - start_frame (int): The starting frame index for multi-camera tracking. Default is 0.
    - end_frame (int): The ending frame index for multi-camera tracking. If None, it tracks until the last frame.
                       Default is None.

        
    """
    id_mapping = {}
    camdirs = sorted(camdirs)

    def get_frame(cap, frame_id):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        _, frame = cap.read()

        return frame

    def process_object(frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        bbox_img = frame[y1:y2, x1:x2]
        bbox_img = Image.fromarray(bbox_img)

        return fe.extract_features(bbox_img)

    # Iterate over all bboxes to create DB
    for cam in range(len(bboxes)):
        print(f"Cam: {cam} so camdir: {camdirs[cam]}")

        frame_id, stop_frame = frame_boundaries[cam]
        while frame_id < stop_frame:
            try: 
                print("frame ", frame_id)

                if not frame_id in bboxes[cam]:
                    continue 

                frame_bboxes = bboxes[cam][frame_id]
                frame = get_frame(video_captures[cam], frame_id)

                for bbox in frame_bboxes:
                    track_id = bbox[0]

                    # Do not extract features again if already in DB
                    if dbh.is_track_id_exists(track_id, seq):
                        continue

                    # Extract feature of object in bbox and save to DB
                    feature_vector = process_object(frame, bbox[1:])
                    dbh.insert_feature(track_id, frame_id, seq, cam, feature_vector, False)

            except: 
                print(f"Something went wrong for {frame_id}")

            finally:
                frame_id += 1

    print(f"---------------------------------------")

    
    # Iterate over all cameras to reidentify    
    for cam in range(len(bboxes)):
        print(f"Cam: {cam} so camdir: {camdirs[cam]}")

        frame_id, stop_frame = frame_boundaries[cam]
        bboxes_out = {}

        print(f"Frame ID: {frame_id} stop_frame: {stop_frame}")
        while frame_id < stop_frame:
            try: 
                print("frame ", frame_id)

                if not frame_id in bboxes[cam]:
                    continue 
                
                # Get the bounding boxes in given camera and at certain frame
                frame_bboxes = bboxes[cam][frame_id]

                for bbox in frame_bboxes:
                    track_id = bbox[0]

                    # Perform similarity search if new tracking_id (not in DB)
                    curr_obj = dbh.get_object(track_id, seq)

                    # Check that this object has not be reidentified already
                    if curr_obj and not curr_obj[-1]:
                        curr_fv = np.frombuffer(curr_obj[-2], dtype=np.float32)

                        # TODO: Add offset according to timestamp 
                        # The issue might be that the first time an object is tracked is way before
                        # it appears in another video
                        objects = dbh.get_between_frames_not_in_camera(frame_id - 50, frame_id + 50, camdirs[cam])
                        max_similarity, max_id = 0, -1
                        for obj in objects:
                            obj_fv = np.frombuffer(obj[-2], dtype=np.float32)

                            similarity = cossim(obj_fv, curr_fv)
                            if similarity > max_similarity:
                                max_similarity, max_id = similarity, obj[0]

                        if max_similarity > 0.85:
                            # Mark object as reidentified
                            dbh.update_reid(track_id, seq, True)

                            # Ensure that the objects do not map to each other
                            if max_id in id_mapping and id_mapping[max_id] != track_id:
                                print(f"{track_id} mapped to {max_id} with {max_similarity} confidence")
                                id_mapping[track_id] = max_id


                    # reid = id_mapping[track_id] if track_id in id_mapping else track_id
                    reid = recursive_track_mapping(id_mapping, track_id)
                    if frame_id not in bboxes_out:
                        bboxes_out[frame_id] = []
                    
                    bboxes_out[frame_id].append((frame_id, bbox[1], bbox[2], bbox[3], bbox[4], reid))
            except Exception as error:
                print(f"Something went wrong for {frame_id} - {error}")
            finally:
                frame_id += 1

        camdir = camdirs[cam]
        track_to_file(bboxes_out, os.path.join(dir_path, seq, camdir, det_name_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SORT tracking and save files for given sequence.")
    parser.add_argument("seq", choices=["S01", "S03", "S04"], help="Sequence option (S01, S03, S04)")
    args = parser.parse_args()
    seq = args.seq

    frame_boundaries = load_frame_number(dir_path, seq, det_name)
    camdirs = os.listdir(os.path.join(dir_path, seq))

    video_captures, img, bboxes = load_sequence(dir_path, img_path, seq, det_name=det_name, with_id=True)

    dbh = DatabaseHandler(f'feature_db_{seq}.db')
    fe = FeatureExtractor()

    multicamera_tracking(video_captures, bboxes, seq, dbh, fe, camdirs, frame_boundaries)