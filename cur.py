import cv2
from ultralytics import YOLO
model = YOLO("best.pt")

video_path = "comp23_4.mp4"

cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    success, frame = cap.read()

    if success:

        results = model(frame)
        print(results)

        annotated_frame = results[0].plot()

        #print(annotated_frame)

        cv2.imshow("YOLO", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()