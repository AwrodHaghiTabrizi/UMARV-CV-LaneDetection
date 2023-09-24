# Creating Datasets

## Lane Detection

### Real world

1. Take a high quality video going through a course.
2. Convert the video to .mp4 if necessary.
3. Store the video in DrobBox ".../ML/raw_videos".
4. Place the video in the UMARV-CV repo "/parapeters/input" directory
5. Run the script "/src/scripts/get_frames_from_video.py"
6. Go to https://app.roboflow.com/umarv-cv
7. Click "+ Create New Project"
8. Project Type = "Semantic Segmentation", What Are You Detecting" = "Lanes, Project Name = "real_world/{name_of_dataset}", License = "CC BY 4.0"
9. Drop all the raw images into roboflow by selecting the folder and pointing to "{UMARV-CV repo}/parameters/output/{dataset_name}/data"
10. Once the raw images are exported to roboflow, delete the contents of input and output in the repo.
11. Click Save and Continue
![Alt text](image.png | width=100)
12. Add teammates if necessary, max of 3 people including yourself.
13. Click "Assign Images"
14. Label the images in the Unannotated section
    1. Use the smart polygon feature.
    2. Click on a lane. Every time you click, it auto detects the lane you clicked on. Add more clicks in other areas of the lane if necessary until the lane is fully covered.
    3. Click enter twice.
    - Tip: You dont need to label every lane at once. You can do them one at a time. Just hit enter once the focused lane is done then shift focus to covering another lane.
    - Tip: If after you click on the lane, more than the lane is covered, click on any part of the over covered area and it will remove that portion.
    - Tip: If you made a mistake on a lane, you can click on "Layers" and delete the Lanes label that was wrong.
    - Tip: For small lanes that are hard to pick up by the smart polygon, you can use the regular polygoon tool.
    - Tip: Pay close attention to detail. Even if they are far or small or in between cracks, any lane you recognize, our algorithms and models should too.
15. Once finished annotating all the images. Go through them again for quality assurance to ensure nothing was missed or incorrect.
16. Click "Add x images to Dataset"
17. Make it 100% Train
18. Click "Add Images"
19. Go to https://app.roboflow.com/umarv-cv
20. Click on the project
21. Remove the preprocessing steps in section 3 and click "Continue"
22. Leave Augmentation empty and click "Continue"
23. Click "Generate"
24. Click "Export Dataset"
25. Format = "Semantic Segmentation Masks"
26. Click "download zip to computer"
27. Click "Continue"
28. Unzip the downloaded folder
29. Rename the "train" folder to "label"
30. Move the label folder into the UMARV-CV repo in the "/parapeters/input" directory. Should be "/parameters/input/label/..."
31. Run the script "/src/scripts/extract_label_from_roboflow.py"
32. Extract the labels from "/parapeters/output"
32. Create a dataset directory with the following folder structure <br>
dataset_name/ <br>
├── data/ <br>
│ ├── 000000.jpg <br>
│ ├── 000001.jpg <br>
│ ├── 000002.jpg <br>
│ └── ... <br>
├── label/ <br>
│ ├── 000000.jpg <br>
│ ├── 000001.jpg <br>
│ ├── 000002.jpg <br>
│ └── ...
33. Export this dataset into the UMARV CV Dropbox in ".../ML/datasets/real_world".

### Unity

...

## Scene Segmentation

...