import cv2
import numpy as np
from YOLOByteTracker import YOLOByteTracker

class Equipment:
    """
    Represents a piece of tracked equipment in a video frame.

    Attributes:
        track_id (int): Unique identifier from the tracker (e.g., ByteTrack ID).
        bbox (list[int]): Bounding box coordinates [x1, y1, x2, y2].
        cls_name (str): Class name of the equipment (e.g., 'excavator').
    """

    def __init__(self, track_id, bbox, cls_name):
        """
        Initialize an Equipment instance.

        Args:
            track_id (int): Tracker ID for this object.
            bbox (list[int]): Bounding box [x1, y1, x2, y2].
            cls_name (str): Class name of the equipment.
        """
        self.track_id = track_id
        self.bbox = bbox  # Bounding box [x1, y1, x2, y2]
        self.cls_name = cls_name

    def is_moving(self, motion_value, threshold):
        """
        Determine if the equipment is moving based on a motion value.

        Args:
            motion_value (float): Motion magnitude (e.g., from optical flow).
            threshold (float): Motion threshold to consider as "moving".

        Returns:
            bool: True if motion_value exceeds threshold, else False.
        """
        return motion_value > threshold    

    def split_regions(self):
        """
        Split the equipment bounding box into predefined regions.

        Regions (top to bottom):
            - bucket: top 15%
            - arm: 15% to 40%
            - body: 40% to 70%
            - tracks: 70% to bottom

        Returns:
            dict: Region name -> bounding box tuple (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = self.bbox
        width = x2 - x1
        height = y2 - y1

        regions = {
            "bucket": (x1, y1, x2, y1 + int(0.15 * height)),
            "arm":    (x1, y1 + int(0.15 * height), x2, y1 + int(0.40 * height)),
            "body":   (x1, y1 + int(0.40 * height), x2, y1 + int(0.70 * height)),
            "tracks": (x1, y1 + int(0.70 * height), x2, y2)
        }

        return regions

    def get_region_image(self, frame, region_box):
        """
        Crop the image to a specific region of the equipment.

        Args:
            frame (numpy.ndarray): Full video frame (BGR).
            region_box (tuple[int]): Region bounding box (x1, y1, x2, y2).

        Returns:
            numpy.ndarray: Cropped image of the region.
        """
        x1, y1, x2, y2 = region_box
        return frame[y1:y2, x1:x2]

    def optical_flow(self, prev_region, curr_region, threshold=1.0):
        """
        Compute average optical flow magnitude between two image regions.

        Args:
            prev_region (numpy.ndarray): Previous region crop (BGR).
            curr_region (numpy.ndarray): Current region crop (BGR).
            threshold (float): Threshold to consider motion as significant.

        Returns:
            tuple:
                - avg_mag (float): Average optical flow magnitude.
                - is_moving (bool): True if avg_mag > threshold, else False.
        """
        # Convert images to grayscale for optical flow
        prev_gray = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_region, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )


        # Compute flow magnitude
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        avg_mag = np.mean(mag)

        # Return average magnitude and movement boolean
        return avg_mag, avg_mag > threshold
    def distance_to(self, other):
     x1, y1, x2, y2 = self.bbox
     ox1, oy1, ox2, oy2 = other.bbox

     cx = (x1 + x2) / 2
     cy = (y1 + y2) / 2
     ocx = (ox1 + ox2) / 2
     ocy = (oy1 + oy2) / 2

     return ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5


   
    def analyze(self, prev_frame, curr_frame, equipment_objects):
        raise NotImplementedError