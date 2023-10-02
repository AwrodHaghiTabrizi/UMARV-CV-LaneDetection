import os
import shutil
import cv2

def main():

    repo_dir = os.getcwd()
    input_dir = f"{repo_dir}/parameters/input"
    output_dir = f"{repo_dir}/parameters/output"

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    if len(video_files) == 0:
        print("Input directory does not contain video.")
        return
    elif len(video_files) > 1:
        print("Input directory contains multiple videos. Should only contain one video.")
        return
    elif len(video_files) == 1:
        dataset_name = video_files[0].replace(".mp4", "")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_dataset_dir = f"{output_dir}/{dataset_name}"
    output_data_dir = f"{output_dataset_dir}/data"
    os.makedirs(output_dataset_dir)
    os.makedirs(output_data_dir)

    video_dir = f"{input_dir}/{video_files[0]}"
    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = 10
    frame_step = int(round(fps/frame_rate))

    frame_index = 0
    data_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_step == 0:
            frame_output_dir = f"{output_dir}/{dataset_name}/data/{data_index:06d}.jpg"
            cv2.imwrite(frame_output_dir, frame)
            data_index += 1

        frame_index += 1

    cap.release()


if __name__ == "__main__":
    main()