# UMARV-CV-LaneDetection

Repository for UMARV Computer Vision for the purpose of solving lane detection.

## Models vs Algorithms

The models folder hosts all of our machine learning solutions, while the algorithms folder hosts our hard coded solutions. Each model/algorithm is seperated into its own folder and has its own unique ID.

## Scripts

The src/scripts folder hosts our scripts which provide varrying functionalities from model/algorithm initialization, performance comparison, and dataset generation. To run them, right click on the script and select "Run Python File in Terminal".

## How To Interact With This Repository

1. Make sure you have git installed on your computer, guide [here](https://git-scm.com/downloads)
2. Get access to the GitHub repository from a team lead
3. Accept the invitation to the GitHub repository
4. Go to https://github.com/AwrodHaghiTabrizi/UMARV-CV-LaneDetection/branches
5. Click "New Branch"
6. Branch name : "users/{your_initials}"
7. Source : main
8. Click "Create Branch"
9. On your PC, create a UMARV folder <br>
├── UMARV/ <br>
10. Create a Lane Detection folder <br>
├── UMARV/ <br>
│ ├── LaneDetection/ <br>
11. Right click into the LaneDetection folder
12. Open a new terminal.
13. In the terminal, type "git clone -b {your_branch_name} https://github.com/AwrodHaghiTabrizi/UMARV-CV-LaneDetection.git"
14. Open VSCode
15. Click File > Open Folder
16. Open the UMARV-CV-LaneDetection folder
- Tip: Keep your working directory as UMARV-CV-LaneDetection when running scripts or notebooks

### Repository Rules

- Full freedom to create/delete/edit code in your model/algorithm folder.
- Dont change any code in:
    - model/algorithm folders that dont belong to you
    - src/script (unless making updates)
    - model_template/algorithm_tempalte (unless making updates)
- Work in your own branch. Pull before every work session. Push after every work session. [git tutorial](https://www.w3schools.com/git/git_intro.asp?remote=github)

## Environments

This repository allows development flexability to work in multiple environments, including: Windows, Mac, Google Colab, and LambdaLabs.
- [Working with Google Colab](https://github.com/AwrodHaghiTabrizi/UMARV-CV-LaneDetection/blob/users/AHT/docs/working_with_environments.md#google-colab)
- [Working with LambdaLabs](https://github.com/AwrodHaghiTabrizi/UMARV-CV-LaneDetection/blob/users/AHT/docs/working_with_environments.md#lambdalabs)

## Developing Models

1. Navigate to src/scripts
2. Right click on either "create_new_model.py" or "create_copy_of_model.py"
3. Click "Run Python File in Termainl"
4. Answer the prompts in the terminal
5. Go through [Working With Models](https://github.com/AwrodHaghiTabrizi/UMARV-CV-LaneDetection/blob/users/AHT/docs/creating_models.md)

## Developing Algorithms

1. Navigate to src/scripts
2. Right click on either "create_new_algorithm.py" or "create_copy_of_algorithm.py"
3. Click "Run Python File in Termainl"
4. Answer the prompts in the terminal
5. Go through [Working With Algorithms](https://github.com/AwrodHaghiTabrizi/UMARV-CV-LaneDetection/blob/users/AHT/docs/creating_algorithms.md)