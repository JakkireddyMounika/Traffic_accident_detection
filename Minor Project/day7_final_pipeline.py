from ultralytics import YOLO
import cv2
import math
import time
from datetime import datetime

# 🔹 MOCK LLM FUNCTION (SAFE FOR DEMO)
def llm_generate_sos(accident_data):
    return (
        "🚨 EMERGENCY ALERT 🚨\n"
        f"Traffic accident detected.\n"
        f"Camera ID: {accident_data['camera_id']}\n"
        f"Time: {accident_data['time']}\n"
        f"Vehicles Involved: {accident_data['vehicles_involved']}\n"
        f"Severity: {accident_data['severity']}\n"
        "Immediate police and medical assistance required."
    )

# 🔹 INITIAL SETUP
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("data/videos/traffic.mp4")

prev_positions = {}
prev_time = time.time()
CAMERA_ID = "CAMERA_01"
alert_sent = False   # avoid repeated alerts

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

# 🔁 MAIN LOOP
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
    involved = set()

    for box in boxes:
        if box.id is None:
            continue

        vid = int(box.id)
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        if vid in prev_positions:
            px, py = prev_positions[vid]
            speed = math.hypot(cx-px, cy-py) / max(time_diff, 1e-5)
            speeds[vid] = speed

        prev_positions[vid] = (cx, cy)

    accident = False
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            iou = calculate_iou(
                boxes[i].xyxy[0].tolist(),
                boxes[j].xyxy[0].tolist()
            )

            id1 = int(boxes[i].id) if boxes[i].id else None
            id2 = int(boxes[j].id) if boxes[j].id else None

            if iou > 0.3 and (speeds.get(id1, 100) < 5 or speeds.get(id2, 100) < 5):
                accident = True
                involved.add(id1)
                involved.add(id2)

    if accident and not alert_sent:
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        accident_data = {
            "camera_id": CAMERA_ID,
            "time": timestamp,
            "vehicles_involved": list(involved),
            "severity": "HIGH"
        }

        sos = llm_generate_sos(accident_data)
        print("\n", sos)
        alert_sent = True

    if accident:
        cv2.putText(
            annotated_frame,
            "ACCIDENT DETECTED - SOS SENT",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("FINAL TRAFFIC ACCIDENT SYSTEM", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
