
### 📝Project Report

```markdown
# Player Re-Identification and Tracking – Report

## 🎯 Objective

Detect and assign unique IDs to each player in a 15-second video, ensuring that even if they exit and re-enter the frame, their identity is preserved.


## ⚙️ Methodology

### Step 1: Object Detection with YOLOv11

- Used the provided YOLOv11 fine-tuned model to detect players in each frame.
- Applied a confidence threshold of 0.4 to reduce false positives.

### Step 2: Deep SORT for Tracking

- Integrated Deep SORT with Kalman Filtering and appearance embedding matching.
- Initial IDs are assigned within the first few frames.
- When a player disappears and reappears, their identity is recovered using cosine similarity on embeddings.


## 🧪 Techniques Used

| Component | Tool |
|----------|------|
| Object Detection | YOLOv11 |
| Multi-Object Tracking | Deep SORT |
| Video Processing | OpenCV |
| ID Assignment | Kalman Filter + ReID Embeddings |


## ✅ Outcomes

- The tracker maintains consistent IDs even after temporary occlusion or exit.
- Real-time simulation achieved by processing frame-by-frame without batch lookahead.


## 🚧 Challenges

- **Model format compatibility:** The original model had to be in YOLOv11 format accepted by `ultralytics`.
- **Appearance similarity issues:** In case of very similar uniforms, embedding misassignments could occur.


## 🔄 Improvements With More Time

- Integrate a custom ReID network (e.g., OSNet) for better embedding discrimination.
- Use ensemble IoU + appearance matching for more robust data association.
- Improve occlusion handling with temporal smoothing.

