import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("best.pt")

tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

cap = cv2.VideoCapture("15sec_input_720p.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("Output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

def xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

while cap.isOpened():
    ret, frame= cap.read()
    if not ret:
        break

    detections = []
    results = model(frame)[0]
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls = box
        if score < 0.4:
            continue
        cls_id = int(cls)
        if(cls_id == 0):
            bbox = xyxy_to_xywh([x1, y1, x2, y2])
            detections.append((bbox, score, 'player'))
    
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'Player {track_id}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    out.write(frame)

cap.release()
out.release()
print("Tracking complete. Output saved as Output.mp4")