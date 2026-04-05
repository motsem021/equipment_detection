from optical_flow import Equipment

class Excavator(Equipment):
    """
    Represents an excavator, subclass of Equipment, with activity classification.

    Activities detected:
        - DIGGING
        - SWINGING_LOADING
        - DUMPING
        - WAITING
        - ACTIVE_UNKNOWN

    Attributes:
        arm_threshold (float): Optical flow threshold for arm movement.
        bucket_threshold (float): Optical flow threshold for bucket movement.
        body_threshold (float): Optical flow threshold for body movement.
        track_threshold (float): Optical flow threshold for tracks movement.
    """

    def __init__(self, track_id, bbox):
        """
        Initialize an Excavator instance with motion thresholds.

        Args:
            track_id (int): Tracker ID for this object.
            bbox (list[int]): Bounding box [x1, y1, x2, y2].
        """
        super().__init__(track_id, bbox, "excavator")

        # Motion thresholds for different parts
        self.arm_threshold = 1.5
        self.bucket_threshold = 1.5
        self.body_threshold = 1.0
        self.track_threshold = 1.0

    def digging(self, arm_motion, bucket_motion, body_motion, track_motion, truck_near=False):
        """
        Determine if the excavator is digging.

        Conditions:
            - Arm and bucket moving
            - Body and tracks not moving
            - Truck not nearby
        """
        arm_moving = self.is_moving(arm_motion, self.arm_threshold)
        bucket_moving = self.is_moving(bucket_motion, self.bucket_threshold)
        body_moving = self.is_moving(body_motion, self.body_threshold)
        track_moving = self.is_moving(track_motion, self.track_threshold)

        return (
            arm_moving
            and bucket_moving
            and not body_moving
            and not track_moving
            and not truck_near
        )

    def swinging_loading(self, arm_motion, bucket_motion, body_motion, track_motion, truck_near=False):
        """
        Detect swinging/loading activity:
            - Arm and body moving
            - Tracks not moving
            - Truck nearby
        """
        arm_moving = self.is_moving(arm_motion, self.arm_threshold)
        bucket_moving = self.is_moving(bucket_motion, self.bucket_threshold)
        body_moving = self.is_moving(body_motion, self.body_threshold)
        track_moving = self.is_moving(track_motion, self.track_threshold)

        return (
            arm_moving
            and body_moving
            and not track_moving
            and truck_near
        )

    def dumping(self, arm_motion, bucket_motion, body_motion, track_motion, truck_near=False):
        """
        Detect dumping activity:
            - Bucket moving
            - Truck nearby
            - Body and tracks not moving
        """
        arm_moving = self.is_moving(arm_motion, self.arm_threshold)
        bucket_moving = self.is_moving(bucket_motion, self.bucket_threshold)
        body_moving = self.is_moving(body_motion, self.body_threshold)
        track_moving = self.is_moving(track_motion, self.track_threshold)

        return (
            bucket_moving
            and truck_near
            and not track_moving
            and not body_moving
        )

    def waiting(self, arm_motion, bucket_motion, body_motion, track_motion):
        """
        Determine if the excavator is idle (all parts stationary).
        """
        arm_moving = self.is_moving(arm_motion, self.arm_threshold)
        bucket_moving = self.is_moving(bucket_motion, self.bucket_threshold)
        body_moving = self.is_moving(body_motion, self.body_threshold)
        track_moving = self.is_moving(track_motion, self.track_threshold)

        return not arm_moving and not bucket_moving and not body_moving and not track_moving

    def classify_activity(self, arm_motion, bucket_motion, body_motion, track_motion, truck_near=False):
        """
        Classify the current activity of the excavator based on motion.

        Returns:
            str: One of DIGGING, SWINGING_LOADING, DUMPING, WAITING, ACTIVE_UNKNOWN
        """
        if self.waiting(arm_motion, bucket_motion, body_motion, track_motion):
            self.current_activity = "WAITING"

        elif self.digging(arm_motion, bucket_motion, body_motion, track_motion, truck_near):
            self.current_activity = "DIGGING"

        elif self.dumping(arm_motion, bucket_motion, body_motion, track_motion, truck_near):
            self.current_activity = "DUMPING"

        elif self.swinging_loading(arm_motion, bucket_motion, body_motion, track_motion, truck_near):
            self.current_activity = "SWINGING_LOADING"

        else:
            self.current_activity = "ACTIVE_UNKNOWN"

        return self.current_activity
    
    def analyze(self, prev_frame, curr_frame, equipment_objects):
        regions = self.split_regions()

        prev_bucket = self.get_region_image(prev_frame, regions["bucket"])
        curr_bucket = self.get_region_image(curr_frame, regions["bucket"])

        prev_arm = self.get_region_image(prev_frame, regions["arm"])
        curr_arm = self.get_region_image(curr_frame, regions["arm"])

        prev_body = self.get_region_image(prev_frame, regions["body"])
        curr_body = self.get_region_image(curr_frame, regions["body"])

        prev_tracks = self.get_region_image(prev_frame, regions["tracks"])
        curr_tracks = self.get_region_image(curr_frame, regions["tracks"])

        bucket_motion, _ = self.optical_flow(prev_bucket, curr_bucket)
        arm_motion, _ = self.optical_flow(prev_arm, curr_arm)
        body_motion, _ = self.optical_flow(prev_body, curr_body)
        track_motion, _ = self.optical_flow(prev_tracks, curr_tracks)

        truck_near = any(
                 other.cls_name == "truck" and self.distance_to(other) < 200
                for other in equipment_objects.values())

        return self.classify_activity(
            arm_motion,
            bucket_motion,
            body_motion,
            track_motion,
            truck_near
        )


class Truck(Equipment):
    """
    Represents a truck, subclass of Equipment, with activity classification.

    Activities detected:
        - LOADING
        - WAITING
        - MOVING
    """

    def __init__(self, track_id, bbox):
        """
        Initialize a Truck instance with body motion threshold.

        Args:
            track_id (int): Tracker ID.
            bbox (list[int]): Bounding box [x1, y1, x2, y2].
        """
        super().__init__(track_id, bbox, "truck")
        self.body_threshold = 1.0  # Threshold for truck movement

    def moving(self, body_motion):
        """
        Determine if the truck is moving.
        """
        return self.is_moving(body_motion, self.body_threshold)

    def waiting(self, body_motion):
        """
        Determine if the truck is idle (not moving).
        """
        return not self.moving(body_motion)

    def loading(self, body_motion, excavator_near=False):
        """
        Determine if the truck is being loaded by an excavator.

        Conditions:
            - Excavator nearby
            - Truck stationary
        """
        body_moving = self.moving(body_motion)
        return excavator_near and not body_moving

    def classify_activity(self, body_motion, excavator_near=False):
        """
        Classify current truck activity based on motion and nearby excavator.

        Returns:
            str: One of LOADING, WAITING, MOVING
        """
        if self.loading(body_motion, excavator_near):
            self.current_activity = "LOADING"
        elif self.waiting(body_motion):
            self.current_activity = "WAITING"
        else:
            self.current_activity = "MOVING"

        return self.current_activity
   
    def analyze(self, prev_frame, curr_frame, equipment_objects):
        regions = self.split_regions()

        prev_body = self.get_region_image(prev_frame, regions["body"])
        curr_body = self.get_region_image(curr_frame, regions["body"])

        body_motion, _ = self.optical_flow(prev_body, curr_body)

        excavator_near = any(
            other.cls_name == "excavator" and self.distance_to(other) < 250
            for other in equipment_objects.values())

        return self.classify_activity(body_motion, excavator_near)