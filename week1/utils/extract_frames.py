import cv2
import os

def extract_frames(video_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    current_frame = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break  # Break the loop if there are no frames to read

        # Save the frame as an image file
        frame_name = os.path.join(output_folder, f"frame_{current_frame:04d}.jpg")
        cv2.imwrite(frame_name, frame)
        # Optionally, print out the frame being saved
        # print(f"Extracted {frame_name}")

        current_frame += 1

    # Release the video capture object
    cap.release()
    print("Finished extracting frames.")
