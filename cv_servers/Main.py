import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from YOLOByteTracker import YOLOByteTracker
from activites import Excavator, Truck
from heatmap import MotionHeatmap
from visualization import draw_equipment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

VIDEO_PATH = os.path.join(PROJECT_DIR, "videos", "input.mp4")
OUTPUT_ACTIVITY = os.path.join(PROJECT_DIR, "videos", "output_equipment_activity.mp4")
OUTPUT_HEATMAP = os.path.join(PROJECT_DIR, "videos", "output_motion_heatmap.mp4")

CLASS_ID_TO_NAME = {
    0: "excavator",
    1: "truck",
}


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_time = 1.0 / fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_activity = cv2.VideoWriter(OUTPUT_ACTIVITY, fourcc, fps, (width, height))
    out_heatmap = cv2.VideoWriter(OUTPUT_HEATMAP, fourcc, fps, (width, height))

    tracker = YOLOByteTracker()
    equipment_objects = {}
    prev_frame = None
    motion_heatmap = None
    frame_count = 0

    print(f"Processing video: {VIDEO_PATH}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} fps")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")

        raw_frame = frame.copy()

        if motion_heatmap is None:
            motion_heatmap = MotionHeatmap(raw_frame.shape)

        tracked_objects = tracker.predict_frame(raw_frame)

        for obj in tracked_objects:
            track_id = obj.get("track_id", -1)
            if track_id == -1:
                continue

            class_id = obj.get("class")
            cls_name = CLASS_ID_TO_NAME.get(class_id)
            if cls_name is None:
                continue

            bbox = obj.get("bbox")
            if bbox is None:
                continue

            if track_id not in equipment_objects:
                if cls_name == "excavator":
                    eq = Excavator(track_id, bbox)
                else:
                    eq = Truck(track_id, bbox)

                eq.cls_name = cls_name
                eq.active_time = 0.0
                eq.idle_time = 0.0
                eq.total_time = 0.0

                equipment_objects[track_id] = eq

            equipment_objects[track_id].bbox = bbox

        if prev_frame is None:
            prev_frame = raw_frame.copy()
            continue

        for eq in equipment_objects.values():
            if eq.bbox is None:
                continue

            try:
                activity = eq.analyze(prev_frame, raw_frame, equipment_objects)
            except Exception as e:
                print(f"[ERROR] analyze() failed for ID {eq.track_id}: {e}")
                activity = "UNKNOWN"

            eq.total_time += frame_time
            if activity in ["WAITING", "IDLE", "INACTIVE"]:
                eq.idle_time += frame_time
            else:
                eq.active_time += frame_time

            motion_heatmap.update(prev_frame, raw_frame, eq.bbox)
            draw_equipment(frame, eq, prev_frame, raw_frame, equipment_objects)

        heatmap_display = motion_heatmap.draw(frame.copy(), alpha=0.45)

        out_activity.write(frame)
        out_heatmap.write(heatmap_display)

        prev_frame = raw_frame.copy()

    cap.release()
    out_activity.release()
    out_heatmap.release()

    print(f"\nDone! Processed {frame_count} frames total.")
    print(f"Output saved to:")
    print(f"  Activity: {OUTPUT_ACTIVITY}")
    print(f"  Heatmap:  {OUTPUT_HEATMAP}")

    if equipment_objects:
        print("\n=== Equipment Utilization Summary ===")
        for eq in equipment_objects.values():
            util = (eq.active_time / max(eq.total_time, 1e-6)) * 100
            print(f"  ID {eq.track_id} ({eq.cls_name}): "
                  f"Active={eq.active_time:.1f}s, Idle={eq.idle_time:.1f}s, "
                  f"Utilization={util:.1f}%")


if __name__ == "__main__":
    main()
