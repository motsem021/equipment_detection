
import cv2
import numpy as np


class MotionHeatmap:
    def __init__(self, frame_shape, decay=0.98, blur_size=31):
        """
        frame_shape: shape of original frame (H, W, C)
        decay: lower -> older motion disappears faster
        blur_size: smoothing for nicer heatmap
        """
        self.heat = np.zeros(frame_shape[:2], dtype=np.float32)
        self.decay = decay
        self.blur_size = blur_size

    def update(self, prev_frame, curr_frame, bbox):
        """
        Add motion inside bbox to the heatmap.

        bbox = (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Keep bbox inside frame
        h, w = self.heat.shape
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return

        prev_roi = prev_frame[y1:y2, x1:x2]
        curr_roi = curr_frame[y1:y2, x1:x2]

        if prev_roi.size == 0 or curr_roi.size == 0:
            return

        prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

        # Motion difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, motion_mask = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)

        # Decay previous heat slowly
        self.heat *= self.decay

        # Add new motion
        self.heat[y1:y2, x1:x2] += motion_mask.astype(np.float32)

    def draw(self, frame, alpha=0.5):
        """
        Overlay heatmap on frame.
        """
        heat = cv2.GaussianBlur(self.heat, (self.blur_size, self.blur_size), 0)

        if heat.max() > 0:
            heat_norm = np.uint8(255 * heat / heat.max())
        else:
            heat_norm = np.uint8(heat)

        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        return cv2.addWeighted(frame, 1 - alpha, heat_color, alpha, 0)
