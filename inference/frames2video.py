import cv2
import os
import subprocess

def frames_to_video(frames_folder, output_video_path, fps=30):
    # Get the list of frame files and sort them by name
    frame_files = [f for f in os.listdir(frames_folder) if f.endswith(".png")]
    if not frame_files:
        raise ValueError(f"No frames found in {frames_folder}. Please check the folder.")
    frame_files.sort()  # Ensure they are sorted in order, such as frame_00000.png, frame_00001.png

    # Read the first frame to determine the width and height of the video
    first_frame_path = os.path.join(frames_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape

    # Initialize the video encoder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v encoding
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)  # Write the frame to the video

    # Release the video file
    video.release()
    print(f"Frames from {frames_folder} have been combined into a video and saved as {output_video_path}")

if __name__ == '__main__':
    # Use the function to combine frames into a video
    frames_folder = '/root/autodl-tmp/prompt_results'  # Folder containing the video frames
    output_video_path = '/root/autodl-tmp/prompt_visual.mp4'  # Path to the output video file

    # Combine frames into a video
    frames_to_video(frames_folder, output_video_path, fps=2)  # fps=60
