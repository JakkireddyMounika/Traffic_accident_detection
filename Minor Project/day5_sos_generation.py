from ultralytics import YOLO
import cv2
import math
import time
from datetime import datetime

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("data/videos/traffic.mp4")

prev_positions = {}
prev_time = time.time()

LOCATION = "SRM Main Gate, Chennai"
CAMERA_ID = "CAMERA_01"


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[1]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    speeds = {}
    involved_vehicles = set()

    for box in boxes:
        if box.id is None:
            continue

        vehicle_id = int(box.id)
        x1, y1, x2, y2 = box.xyxy[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if vehicle_id in prev_positions:
            px, py = prev_positions[vehicle_id]
            distance = math.hypot(cx - px, cy - py)
            speed = distance / max(time_diff, 1e-5)
            speeds[vehicle_id] = speed

        prev_positions[vehicle_id] = (cx, cy)

    accident_detected = False

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(
                boxes[i].xyxy[0].tolist(),
                boxes[j].xyxy[0].tolist()
            )

            id1 = int(boxes[i].id) if boxes[i].id is not None else None
            id2 = int(boxes[j].id) if boxes[j].id is not None else None

            speed1 = speeds.get(id1, 100)
            speed2 = speeds.get(id2, 100)

            if iou > 0.3 and (speed1 < 5 or speed2 < 5):
                accident_detected = True
                involved_vehicles.add(id1)
                involved_vehicles.add(id2)

    if accident_detected:
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        # 🔹 STRUCTURED ACCIDENT DATA
        accident_data = {
            "camera_id": CAMERA_ID,
            "location": LOCATION,
            "time": timestamp,
            "vehicles_involved": list(involved_vehicles),
            "severity": "HIGH"
        }

        # 🔹 SOS MESSAGE (LLM-STYLE OUTPUT)
        sos_message = (
            f"🚨 TRAFFIC ACCIDENT ALERT 🚨\n"
            f"Camera ID: {accident_data['camera_id']}\n"
            f"Location: {accident_data['location']}\n"
            f"Time: {accident_data['time']}\n"
            f"Vehicles Involved: {accident_data['vehicles_involved']}\n"
            f"Severity: {accident_data['severity']}\n"
            f"Immediate emergency response required."
        )

        print("\n", sos_message)

        cv2.putText(
            annotated_frame,
            "SOS GENERATED",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Day-5 SOS Generation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
