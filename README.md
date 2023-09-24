# UMARV-CV-LaneDetection

Repository for UMARV Computer Vision for the purpose of solving lane detection.

## Models vs Algorithms

The models folder hosts all of our machine learning solutions, while the algorithms folder hosts our hard coded solutions. Each model/algorithm is seperated into its own folder and has its own unique ID.

## Scripts

The src/scripts folder hosts our scripts which provide varrying functionalities from model/algorithm initialization, performance comparison, and dataset generation. To run them, right click on the script and select "Run Python File in Terminal"

## How To Interact With This Repository

1. Make sure you have git installed on your computer, guide [here](youtube.com)
2. Go to https://github.com/AwrodHaghiTabrizi/UMARV-CV-LaneDetection/branches
3. Click "New Branch"
4. Branch name : "users/{your_initials}"
5. Source : main
6. Click "Create Branch"
7. Create a UMARV folder <br>
├── UMARV/ <br>
8. Create a Lane Detection folder <br>
├── UMARV/ <br>
│ ├── LaneDetection/ <br>
9. Right click into the LaneDetection folder
10. Click "Git Bash Here"
11. In the terminal, type "git clone -b {your_branch_name} https://github.com/AwrodHaghiTabrizi/UMARV-CV-LaneDetection.git"

### Repository Rules

- Full freedom to create/delete/edit code in your model/algorithm folder.
- Dont change any code in:
    - model/algorithm folders that dont belong to you
    - src/script (unless making updates)
    - model_template/algorithm_tempalte (unless making updates)
- Work in your own branch. Pull before every work session. Push after every work session. [git tutorial](youtube.com)

## Environments

This repository allows development flexability to work in multiple environments, including: Windows, Mac, Google Colab, and LambdaLabs.
- [Working in Google Colab](youtube.com)
- [Working in LambdaLabs](youtube.com)

## Developing Models

1. Navigate to src/scripts
2. Right click on either "create_model.py" or "create_copy_of_model.py"
3. Click "Run Python File in Termainl"
4. Answer the prompts in the terminal
5. Go through [Working With Models](youtube.com)

## Developing Algorithms

1. Navigate to src/scripts
2. Right click on either "create_algorithm.py" or "create_copy_of_algorithm.py"
3. Click "Run Python File in Termainl"
4. Answer the prompts in the terminal
5. Go through [Working With Algorithms](youtube.com)