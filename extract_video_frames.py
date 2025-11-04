import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    vidcap = cv2.VideoCapture(video_path)

    frame_count = 0
    success, image = vidcap.read()
    while success:
        # Save each frame as JPEG file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, image)

        frame_count += 1
        success, image = vidcap.read()

    print(f"Extracted {frame_count} frames to folder: {output_folder}")

if __name__ == "__main__":
    video_file = "lion_video.mp4"         # Path to your lion video file
    output_dir = "lion_imgs"        # Folder to save extracted images
    extract_frames(video_file, output_dir)
