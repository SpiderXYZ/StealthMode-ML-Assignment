# Player Re-Identification and Tracking

## 🎯 Objective
Given a 15-second video clip, detect all players and ensure each player maintains the same ID even if they exit and re-enter the frame.

---

## Setup Instructions

### 1. Clone the Repository or Download Files

Place the following files in one directory:
- `track.py`
- `15sec_input_720p.mp4` (input video)
- `best.pt` (detection model)

### 2. Install Dependencies

- ```bash
  pip install opencv-python ultralytics deep_sort_realtime

### 3. Running the Code
- ```bash
    python track.py
  
