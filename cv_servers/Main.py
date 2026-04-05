import cv2
import numpy as np

from YOLOByteTracker import YOLOByteTracker
from activites import Excavator, Truck
from heatmap import MotionHeatmap
from visualization import draw_equipment  # Your vis.py functions

VIDEO_PATH = r"C:\Users\sesra\OneDrive\Desktop\PS\project\equipment_detector\cv_servers\WhatsApp Video 2026-04-04 at 3.53.54 PM.mp4"

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

    tracker = YOLOByteTracker()
    equipment_objects = {}
    prev_frame = None
    motion_heatmap = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frame = frame.copy()

        # Initialize heatmap object once
        if motion_heatmap is None:
            motion_heatmap = MotionHeatmap(raw_frame.shape)

        # Detect and track objects
        tracked_objects = tracker.predict_frame(raw_frame)

        # Register or update equipment objects
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

            # Create equipment instance if new
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

            # Always update current bbox
            equipment_objects[track_id].bbox = bbox

        # Skip first frame (no prev_frame yet)
        if prev_frame is None:
            prev_frame = raw_frame.copy()
            continue

        # Analyze each tracked equipment
        for eq in equipment_objects.values():
            if eq.bbox is None:
                continue

            # Update total_time / activity
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

            # Update heatmap
            motion_heatmap.update(prev_frame, raw_frame, eq.bbox)

            # Draw equipment with proper equipment_objects passed
            draw_equipment(frame, eq, prev_frame, raw_frame, equipment_objects)

        # Create displayable heatmap overlay
        heatmap_display = motion_heatmap.draw(frame.copy(), alpha=0.45)

        # Show results
        cv2.imshow("Equipment Activity", frame)
        cv2.imshow("Motion Heatmap", heatmap_display)

        # Save current frame for next iteration
        prev_frame = raw_frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()