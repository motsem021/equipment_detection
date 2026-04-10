import streamlit as st
import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cv_servers"))

from YOLOByteTracker import YOLOByteTracker
from activites import Excavator, Truck
from heatmap import MotionHeatmap
from visualization import draw_equipment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "input.mp4")

CLASS_ID_TO_NAME = {0: "excavator", 1: "truck"}

ACTIVITY_BADGE = {
    "DIGGING":          ("🟠", "orange"),
    "SWINGING_LOADING": ("🟡", "goldenrod"),
    "DUMPING":          ("🔴", "red"),
    "WAITING":          ("⚪", "gray"),
    "ACTIVE_UNKNOWN":   ("🟢", "green"),
    "LOADING":          ("🟡", "goldenrod"),
    "MOVING":           ("🟢", "green"),
    "UNKNOWN":          ("⚫", "black"),
}

IDLE_ACTIVITIES = {"WAITING", "IDLE", "INACTIVE"}

st.set_page_config(
    page_title="Equipment Utilization Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .machine-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .machine-title {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
    }
    .status-active {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-idle {
        display: inline-block;
        background: #f3f4f6;
        color: #6b7280;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .activity-tag {
        font-size: 0.82rem;
        font-weight: 500;
        color: #374151;
        margin-top: 4px;
    }
    .metric-row {
        display: flex;
        gap: 8px;
        margin-top: 10px;
    }
    .metric-box {
        flex: 1;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f2937;
    }
    .util-bar-bg {
        background: #e5e7eb;
        border-radius: 99px;
        height: 8px;
        margin-top: 6px;
        overflow: hidden;
    }
    .frame-counter {
        font-size: 0.8rem;
        color: #9ca3af;
        text-align: right;
        margin-top: 4px;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)


def fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}m {s:02d}s"


def util_color(pct: float) -> str:
    if pct >= 70:
        return "#10b981"
    if pct >= 40:
        return "#f59e0b"
    return "#ef4444"


def render_machine_card(eq_data: dict) -> str:
    eq_id = eq_data["track_id"]
    cls = eq_data["cls_name"].capitalize()
    activity = eq_data["activity"]
    active_t = eq_data["active_time"]
    idle_t = eq_data["idle_time"]
    total_t = eq_data["total_time"]
    util = (active_t / max(total_t, 1e-6)) * 100
    is_idle = activity in IDLE_ACTIVITIES
    badge_emoji, _ = ACTIVITY_BADGE.get(activity, ("⚫", "black"))
    status_class = "status-idle" if is_idle else "status-active"
    status_text = "INACTIVE" if is_idle else "ACTIVE"
    bar_color = util_color(util)

    return f"""
<div class="machine-card">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span class="machine-title">{'🚜' if cls == 'Excavator' else '🚛'} {cls} #{eq_id}</span>
    <span class="{status_class}">{status_text}</span>
  </div>
  <div class="activity-tag">{badge_emoji} {activity}</div>
  <div class="metric-row">
    <div class="metric-box">
      <div class="metric-label">Working</div>
      <div class="metric-value">{fmt_time(active_t)}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Idle</div>
      <div class="metric-value">{fmt_time(idle_t)}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Utilization</div>
      <div class="metric-value" style="color:{bar_color}">{util:.0f}%</div>
    </div>
  </div>
  <div class="util-bar-bg">
    <div style="width:{min(util,100):.0f}%;background:{bar_color};height:8px;border-radius:99px;transition:width 0.4s;"></div>
  </div>
</div>
"""


def process_video():
    if not os.path.exists(VIDEO_PATH):
        st.error(f"Video not found: `{VIDEO_PATH}`")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        st.error("Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = YOLOByteTracker()
    equipment_objects: dict = {}
    prev_frame = None
    motion_heatmap = None
    frame_count = 0

    col_vid, col_dash = st.columns([2, 1], gap="large")

    with col_vid:
        st.markdown("#### 📹 Live Video Feed")
        tab_activity, tab_heatmap = st.tabs(["Activity View", "Heatmap View"])
        with tab_activity:
            img_placeholder = st.empty()
        with tab_heatmap:
            heat_placeholder = st.empty()
        frame_info = st.empty()

    with col_dash:
        st.markdown("#### 📊 Machine Status & Utilization")
        summary_placeholder = st.empty()
        st.markdown("---")
        machines_placeholder = st.empty()

    stop_btn = st.button("⏹ Stop", key="stop_btn", type="secondary")
    if stop_btn:
        st.session_state["running"] = False
        cap.release()
        return

    while st.session_state.get("running", True):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker = YOLOByteTracker()
            prev_frame = None
            continue

        frame_count += 1
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
                eq = Excavator(track_id, bbox) if cls_name == "excavator" else Truck(track_id, bbox)
                eq.cls_name = cls_name
                eq.active_time = 0.0
                eq.idle_time = 0.0
                eq.total_time = 0.0
                eq.current_activity = "UNKNOWN"
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
            except Exception:
                activity = "UNKNOWN"

            eq.current_activity = activity
            eq.total_time += frame_time
            if activity in IDLE_ACTIVITIES:
                eq.idle_time += frame_time
            else:
                eq.active_time += frame_time

            motion_heatmap.update(prev_frame, raw_frame, eq.bbox)
            draw_equipment(frame, eq, prev_frame, raw_frame, equipment_objects)

        heatmap_display = motion_heatmap.draw(frame.copy(), alpha=0.45)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        heat_rgb = cv2.cvtColor(heatmap_display, cv2.COLOR_BGR2RGB)

        with col_vid:
            with tab_activity:
                img_placeholder.image(frame_rgb, use_container_width=True)
            with tab_heatmap:
                heat_placeholder.image(heat_rgb, use_container_width=True)
            progress = frame_count / max(total_frames, 1)
            frame_info.markdown(
                f'<div class="frame-counter">Frame {frame_count} / {total_frames} &nbsp;|&nbsp; '
                f'Progress: {progress*100:.1f}%</div>',
                unsafe_allow_html=True
            )

        with col_dash:
            active_count = sum(
                1 for eq in equipment_objects.values()
                if getattr(eq, "current_activity", "UNKNOWN") not in IDLE_ACTIVITIES
            )
            total_count = len(equipment_objects)
            summary_placeholder.markdown(
                f"""
                <div style="display:flex;gap:12px;margin-bottom:8px;">
                  <div class="metric-box" style="background:#d1fae5;border-color:#a7f3d0;flex:1;">
                    <div class="metric-label">Active</div>
                    <div class="metric-value" style="color:#065f46;">{active_count}</div>
                  </div>
                  <div class="metric-box" style="background:#f3f4f6;flex:1;">
                    <div class="metric-label">Idle</div>
                    <div class="metric-value" style="color:#6b7280;">{total_count - active_count}</div>
                  </div>
                  <div class="metric-box" style="flex:1;">
                    <div class="metric-label">Total</div>
                    <div class="metric-value">{total_count}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            cards_html = "".join(
                render_machine_card({
                    "track_id": eq.track_id,
                    "cls_name": eq.cls_name,
                    "activity": getattr(eq, "current_activity", "UNKNOWN"),
                    "active_time": eq.active_time,
                    "idle_time": eq.idle_time,
                    "total_time": eq.total_time,
                })
                for eq in sorted(equipment_objects.values(), key=lambda e: e.track_id)
            )
            machines_placeholder.markdown(cards_html, unsafe_allow_html=True)

        prev_frame = raw_frame.copy()

    cap.release()


def main():
    st.markdown('<div class="main-header">🏗️ Equipment Utilization Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time activity classification for heavy machinery using YOLOv8 + ByteTrack + Optical Flow</div>', unsafe_allow_html=True)

    if "running" not in st.session_state:
        st.session_state["running"] = False

    if not st.session_state["running"]:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("▶ Start Processing", type="primary", use_container_width=True):
                st.session_state["running"] = True
                st.rerun()

        st.info("Click **Start Processing** to begin video analysis. The model will detect excavators and trucks, classify their activities, and update the dashboard in real-time.")

        if os.path.exists(os.path.join(BASE_DIR, "videos", "output_equipment_activity.mp4")):
            st.markdown("---")
            st.markdown("#### Previous Run Outputs")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Activity Annotated Video**")
                with open(os.path.join(BASE_DIR, "videos", "output_equipment_activity.mp4"), "rb") as f:
                    st.video(f.read())
            with c2:
                st.markdown("**Motion Heatmap Video**")
                with open(os.path.join(BASE_DIR, "videos", "output_motion_heatmap.mp4"), "rb") as f:
                    st.video(f.read())
    else:
        process_video()
        st.session_state["running"] = False


if __name__ == "__main__":
    main()
