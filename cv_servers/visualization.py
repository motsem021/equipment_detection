import cv2
import numpy as np
from activites import Excavator

# Activity colors (BGR)
ACTIVITY_COLORS = {
    "DIGGING": (0, 165, 255),
    "SWINGING_LOADING": (0, 255, 255),
    "DUMPING": (0, 0, 255),
    "WAITING": (128, 128, 128),
    "ACTIVE_UNKNOWN": (0, 255, 0),
    "LOADING": (0, 255, 255),
    "MOVING": (0, 255, 0)
}

def draw_text(frame, bbox, texts, color, line_height=18):
    x1, y1, x2, y2 = bbox
    for i, text in enumerate(texts):
        y_pos = min(frame.shape[0] - 5, y2 + (i+1) * line_height)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y_pos - h - 2), (x1 + w, y_pos + 2), (0, 0, 0), -1)
        cv2.putText(frame, text, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_equipment(frame, eq, prev_frame, curr_frame, equipment_objects):
    """
    Draw bounding box, activity info, and optical flow arrows for Excavator.
    Pass in equipment_objects so analyze() works.
    """
    x1, y1, x2, y2 = eq.bbox

    # Analyze activity with proper equipment_objects
    activity = eq.analyze(prev_frame, curr_frame, equipment_objects)
    color = ACTIVITY_COLORS.get(activity, (0, 255, 0))

    # Draw bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Text info
    utilization = (eq.active_time / eq.total_time) * 100 if eq.total_time > 0 else 0
    texts = [
        f"ID: {eq.track_id}",
        f"{eq.cls_name} - {activity}",
        f"Active: {eq.active_time:.1f}s  Idle: {eq.idle_time:.1f}s",
        f"Utilization: {utilization:.1f}%"
    ]
    draw_text(frame, eq.bbox, texts, color)

    # Draw motion arrows if Excavator
    if isinstance(eq, Excavator):
        regions = eq.split_regions()
        for part in ["bucket", "arm", "body", "tracks"]:
            prev_img = eq.get_region_image(prev_frame, regions[part])
            curr_img = eq.get_region_image(curr_frame, regions[part])
            if prev_img.size == 0 or curr_img.size == 0:
                continue
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            draw_flow_arrow(frame, regions[part], flow)

def draw_flow_arrow(frame, region_box, flow, scale=5):
    x1, y1, x2, y2 = region_box
    region_flow = flow[y1:y2, x1:x2]
    if region_flow.size == 0:
        return
    avg_dx = int(np.mean(region_flow[..., 0]))
    avg_dy = int(np.mean(region_flow[..., 1]))
    cx, cy = (x1 + x2)//2, (y1 + y2)//2
    cv2.arrowedLine(frame, (cx, cy), (cx + avg_dx*scale, cy + avg_dy*scale), (255, 255, 255), 2)