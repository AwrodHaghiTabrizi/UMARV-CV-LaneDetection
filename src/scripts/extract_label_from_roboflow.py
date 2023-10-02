import os
import shutil
import glob
import cv2
import numpy as np

def main():

    repo_dir = os.getcwd()
    input_dir = f"{repo_dir}/parameters/input"
    output_dir = f"{repo_dir}/parameters/output"

    if not os.listdir(input_dir):
        print("Input directory is empty.")
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_label_dir = f"{output_dir}/label"
    os.makedirs(output_label_dir)

    label_dirs = sorted(glob.glob(f"{input_dir}/label/*_mask.png"))
    idxs = [os.path.basename(label_dir).split("_")[0] for label_dir in label_dirs]

    for idx, label_dir in zip(idxs, label_dirs):
        roboflow_label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
        label = np.zeros_like(roboflow_label)
        label[roboflow_label > 0] = 255
        cv2.imwrite(f"{output_dir}/label/{idx}.jpg", label)

if __name__ == "__main__":
    main()