import numpy as np
import cv2

def create_canvas(frame, videos_number):
    height, width, _ = frame.shape

    # Create a canvas to display all videos
    canvas_width = width
    canvas_height = height
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    rows = 2
    slots_per_row = (videos_number // rows)

    if videos_number % rows != 0:
        slots_per_row += 1 

    width = width // slots_per_row
    height = height // slots_per_row

    # Return canvas, and w and h of one slot
    return canvas, height, width

def show_tracked(cap, tracked, start_frame=1, end_frame=None, mode='show', roi=None):
    frame_id = start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracking_history = {}
    colors = {}

    # Play the video
    while cap.isOpened(): 
        # Capture frame-by-frame 
        ret, frame = cap.read() 

        if not ret or (end_frame is not None and frame_id >= end_frame):
            break

        for bbox in tracked[frame_id]:
            obj_id = int(bbox[0])
            bbox = list(map(int, bbox[1:]))

            # Assign a unique color if new object
            if obj_id not in colors:
                colors[obj_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

            # Draw the bounding box
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            frame = cv2.rectangle(frame, start_point, end_point, colors[obj_id], 2)
            frame = cv2.putText(frame, str(obj_id), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, colors[obj_id], 2, cv2.LINE_AA)
            
            # Update tracking history
            center_position = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
            if obj_id not in tracking_history:
                tracking_history[obj_id] = [center_position]
            else:
                tracking_history[obj_id].append(center_position)
            
            # Draw tracking line (polyline for all historical positions)
            if len(tracking_history[obj_id]) > 1:
                for j in range(1, len(tracking_history[obj_id])):
                    cv2.line(frame, tracking_history[obj_id][j - 1], tracking_history[obj_id][j], colors[obj_id], 2)

            
        if mode == 'show':
            cv2.imshow('Foreground detection', frame)

        frame_id += 1

        # Press Q on keyboard to exit 
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# TODO: Add track id to vis
def show_videos(video_captures, video_order, img, start_frame=0, end_frame=None):    # Read frames from all video captures
    for cap in video_captures:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) 
    
    frame_define = video_captures[0].read()[1]
    curr_frame = start_frame

    canvas, frame_height, frame_width = create_canvas(frame_define, len(video_captures))
    canvas_height, canvas_width, _ = canvas.shape

    while True:
        if end_frame is not None and curr_frame >= end_frame:
            break

        frames = [cap.read()[1] for cap in video_captures]

        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (50, 50) 
        fontScale = 3
        color = (255, 0, 0) 
        thickness = 2
        
        # Resize frames to fit the canvas
        resized_frames = [cv2.putText(frame, str(i + 1), org, font, fontScale, color, thickness, cv2.LINE_AA) for i, frame in enumerate(frames)]
        resized_frames = [cv2.resize(frame, (frame_width, frame_height)) for frame in frames]

        # Arrange frames on the canvas
        start_height, end_height = 0, frame_height
        start_width, end_width = 0, frame_width
        for i in range(len(video_captures)):
            canvas[start_height:end_height, start_width:end_width] = resized_frames[video_order[i]]
            start_height += frame_height
            end_height += frame_height

            if end_height == canvas_height:
                start_height, end_height = 0, frame_height
                start_width += frame_width
                end_width += frame_width
                
        # Display the canvas
        cv2.imshow('Canvas', canvas)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        curr_frame += 1

    # Release resources
    for cap in video_captures:
        cap.release()
    
    cv2.destroyAllWindows()