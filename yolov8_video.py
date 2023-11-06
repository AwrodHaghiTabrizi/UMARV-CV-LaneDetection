from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source = "comp23_4.mp4", show = True, save = True, hide_labels = False, hide_conf=False, conf = 0.5, save_txt = True, save_crop = False, line_thickness = 2, show_conf = True, save_conf = True, vid_stride = True)