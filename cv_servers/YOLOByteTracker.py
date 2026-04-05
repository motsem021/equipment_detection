import os
import cv2
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best (2).pt")


class YOLOByteTracker:
    """
    YOLOByteTracker integrates YOLOv8 object detection with ByteTrack tracking.

    This class allows detecting and tracking objects (e.g., equipment) in video frames.
    It uses the Ultralytics YOLOv8 model with a configurable tracker (default: ByteTrack).

    Attributes:
        model (YOLO): The loaded YOLOv8 model.
        conf (float): Minimum detection confidence threshold.
        tracker_config (str): Tracker configuration file path (YAML), default is ByteTrack.
    """

    def __init__(self,
                 model_path=None,
                 conf=0.5,
                 tracker="bytetrack.yaml"):
        """
        Initialize YOLOByteTracker with a model, confidence threshold, and tracker configuration.

        Args:
            model_path (str): File path to the trained YOLOv8 model (.pt file).
            conf (float): Detection confidence threshold (0.0 - 1.0).
            tracker (str): Path to YOLOv8 tracker YAML configuration. Defaults to ByteTrack.
        """
        if model_path is None:
            model_path = MODEL_PATH
        self.model = YOLO(model_path)
        self.conf = conf
        self.tracker_config = tracker

    def predict_frame(self, frame):
        """
        Detect and track objects on a single video frame.

        Args:
            frame (numpy.ndarray): A single video frame (BGR image).

        Returns:
            list[dict]: A list of tracked objects, each represented as a dictionary:
                {
                    'track_id': int,        # Unique ByteTrack ID for the object
                    'bbox': [x1, y1, x2, y2], # Bounding box coordinates
                    'class': int,           # Detected class ID
                    'confidence': float     # Detection confidence score
                }
        """
        results = self.model.track(frame, persist=True, tracker=self.tracker_config, conf=self.conf)

        tracked_objects = []

        for res in results:
            if len(res.boxes) == 0:
                continue

            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else -1

                tracked_objects.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "class": int(box.cls[0]),
                    "confidence": float(box.conf[0])
                })

        return tracked_objects
