import cv2
import supervision as sv
from ultralytics import YOLO


def main():
    
    # to save the video
    writer= cv2.VideoWriter('webcam_yolo.mp4', 
                            cv2.VideoWriter_fourcc(*'DIVX'), 
                            7, 
                            (1280, 720))
    
    # define resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # specify the model
    model = YOLO("yolov8n.pt")

    # customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
        f"{model.names[confidence]} {class_id: 0.2f}"
        for _, _, class_id,confidence, _
        in detections
        ]


        frame = box_annotator.annotate(
        scene=frame, 
        detections=detections, 
        labels=labels
        ) 

        writer.write(frame)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27): # break with escape key
            break
            
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

