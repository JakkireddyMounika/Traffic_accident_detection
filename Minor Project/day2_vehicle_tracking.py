from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("data/videos/traffic.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    for box in results[0].boxes:
        if box.id is not None:
             print("Vehicle ID:", int(box.id))

    annotated_frame = results[0].plot()
    cv2.imshow("Vehicle Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
