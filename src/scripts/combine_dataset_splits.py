import os
import shutil
import glob

def main():

    repo_dir = os.getcwd()
    input_dir = f"{repo_dir}/parameters/input"
    output_dir = f"{repo_dir}/parameters/output"

    if not os.listdir(input_dir):
        print("Input directory is empty.")
        return

    input_folder_names = []
    for entry in os.listdir(input_dir):
        entry_path = os.path.join(input_dir, entry)
        if os.path.isdir(entry_path):
            input_folder_names.append(entry)
    if len(input_folder_names) == 0:
        print("Input directory does not contain any folders.")
        return
    elif len(input_folder_names) > 1:
        print("Input directory contains multiple folders. Should only contain one folder.")
        return
    elif len(input_folder_names) == 1:
        dataset_name = input_folder_names[0]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_dataset_dir = f"{output_dir}/{dataset_name}"
    output_data_dir = f"{output_dataset_dir}/data"
    output_label_dir = f"{output_dataset_dir}/label"
    os.makedirs(output_dataset_dir)
    os.makedirs(output_data_dir)
    os.makedirs(output_label_dir)

    splits = ["train", "val", "test"]
    index_offset = 0

    for split in splits:
        split_data_dirs = sorted(glob.glob(f"{input_dir}/{dataset_name}/{split}/data/*.jpg"))
        split_label_dirs = sorted(glob.glob(f"{input_dir}/{dataset_name}/{split}/label/*.jpg"))

        if len(split_data_dirs) != len(split_label_dirs):
            print(f"Number of data files {(len(split_data_dirs))} and label files {(len(split_label_dirs))} in {split} split are not the same.")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
            return

        for data_dir, label_dir in zip(split_data_dirs, split_label_dirs):
            shutil.copy(data_dir, f"{output_dir}/{dataset_name}/data/{index_offset:06d}.jpg")
            shutil.copy(label_dir, f"{output_dir}/{dataset_name}/label/{index_offset:06d}.jpg")
            index_offset += 1

if __name__ == "__main__":
    main()