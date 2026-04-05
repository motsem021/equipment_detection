# file: yolo_bytrack_tracker.py
from ultralytics import YOLO
import cv2

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
                 model_path=r"C:\Users\sesra\OneDrive\Desktop\PS\project\equipment_detector\cv_servers\best (2).pt", 
                 conf=0.5, 
                 tracker="bytetrack.yaml"):
        """
        Initialize YOLOByteTracker with a model, confidence threshold, and tracker configuration.

        Args:
            model_path (str): File path to the trained YOLOv8 model (.pt file).
            conf (float): Detection confidence threshold (0.0 - 1.0).
            tracker (str): Path to YOLOv8 tracker YAML configuration. Defaults to ByteTrack.
        """
        self.model = YOLO(model_path)  # Load YOLOv8 model
        self.conf = conf               # Detection confidence threshold
        self.tracker_config = tracker  # Tracker configuration

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
        # Perform tracking on the frame using YOLOv8's built-in tracking API
        results = self.model.track(frame, persist=True, tracker=self.tracker_config, conf=self.conf)
        
        tracked_objects = []

        for res in results:
            # Skip if no objects are detected
            if len(res.boxes) == 0:
                continue

            # Loop through detected boxes
            for box in res.boxes:
                # Extract bounding box coordinates and convert to integers
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Assign a track ID; if not available, default to -1
                track_id = int(box.id[0]) if box.id is not None else -1

                # Append the tracked object information
                tracked_objects.append({
                    "track_id": track_id,     
                    "bbox": [x1, y1, x2, y2],
                    "class": int(box.cls[0]),       
                    "confidence": float(box.conf[0])
                })

        return tracked_objects