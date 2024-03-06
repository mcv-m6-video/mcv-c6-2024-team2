import cv2
import numpy as np

dir_path = 'AICity_data/train/S03/c010/'
video_path = dir_path + 'vdo.avi'
frames_path = 'frames/'

def play_video(cap, frames_path, start_frame=0, end_frame=None, mode='show', roi=None):
    frame_id = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Play the video
    while cap.isOpened(): 
        # Capture frame-by-frame 
        ret, frame = cap.read() 

        if not ret or (end_frame is not None and frame_id >= end_frame):
            break

        if roi is not None:
            roi_coord = np.where(roi == 0)
            frame[roi_coord[0], roi_coord[1]] = 0
            
        if mode == 'show':
            cv2.imshow('Foreground detection', frame)
        elif mode == 'write':
            cv2.imwrite(f'{frames_path}frame{frame_id}.png', frame)

        frame_id += 1

        # Press Q on keyboard to exit 
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap = cv2.VideoCapture(video_path)
play_video(cap, frames_path, mode='write')