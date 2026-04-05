# Equipment Utilization Activity Classification

## Overview

A Python computer vision application that detects, tracks, and classifies the activities of heavy machinery (excavators and trucks) in video footage. It uses YOLOv8 for object detection and ByteTrack for tracking, then analyzes motion using dense optical flow to classify activities like DIGGING, SWINGING_LOADING, DUMPING, WAITING, LOADING, and MOVING.

## Project Structure

```
.
├── cv_servers/
│   ├── Main.py              # Entry point — processes video, writes output files
│   ├── YOLOByteTracker.py   # YOLOv8 + ByteTrack detection/tracking
│   ├── activites.py         # Activity classification (Excavator, Truck classes)
│   ├── optical_flow.py      # Base Equipment class + optical flow utilities
│   ├── heatmap.py           # Motion heatmap generation
│   ├── visualization.py     # Frame annotation (bounding boxes, labels, arrows)
│   ├── db.py                # PostgreSQL integration (optional, not wired in Main.py)
│   ├── kafka.py             # Kafka integration (optional, not wired in Main.py)
│   └── best (2).pt          # Trained YOLOv8 model weights
├── videos/
│   ├── input.mp4            # Input video to process
│   ├── output_equipment_activity.mp4   # Output with activity annotations
│   └── output_motion_heatmap.mp4       # Output with motion heatmap overlay
├── requirements.txt
└── replit.md
```

## How It Works

1. Reads `videos/input.mp4` frame by frame
2. Runs YOLOv8 detection + ByteTrack tracking on each frame
3. For each tracked piece of equipment, computes dense optical flow on sub-regions (bucket, arm, body, tracks for excavators)
4. Classifies the activity based on which regions are moving and proximity to other equipment
5. Writes annotated frames to two output MP4 files
6. Prints an equipment utilization summary at the end

## Running

**Streamlit UI (primary):** `streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true`
- Provides a live dashboard with video feed, machine status, and utilization metrics
- Click "Start Processing" in the UI to begin analysis

**CLI batch mode:** `cd cv_servers && python Main.py`
- Processes the video headlessly and saves output MP4 files

To process a different video, replace `videos/input.mp4` or update `VIDEO_PATH` in `cv_servers/Main.py`.

## Dependencies

- **ultralytics**: YOLOv8 model and ByteTrack tracker
- **opencv-python-headless**: Video I/O and optical flow (headless — no GUI)
- **numpy**: Numerical operations
- **lap**: Linear assignment problem solver (used by ByteTrack)

## Key Design Notes

- The app runs headlessly (no GUI windows) — all output goes to video files and stdout
- Hardcoded Windows paths from the original repo have been replaced with relative paths
- `db.py` and `kafka.py` are provided as integrations but not used by default in `Main.py`
- The model file `best (2).pt` is a custom-trained YOLOv8 model for excavator/truck detection
