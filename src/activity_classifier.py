import numpy as np

class DriverActivityClassifier:
    def __init__(self):
        # COCO keypoint indices
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12

    def classify(self, keypoints: np.ndarray) -> str:
        """
        Classify driver state based on keypoints.
        """
        keypoints = np.array(keypoints, dtype=np.float32).reshape(17, 2)

        if not self._valid_head_and_shoulders(keypoints):
            return "Unknown"

        if self.is_drowsy(keypoints):
            return "Drowsy"
        elif self.is_distracted(keypoints):
            return "Distracted"
        else:
            return "Normal"

    def _valid_head_and_shoulders(self, kpts: np.ndarray) -> bool:
        # Require nose and both shoulders
        required = [self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER]
        return all(not np.allclose(kpts[i], [0.0, 0.0]) for i in required)

    def is_distracted(self, kpts: np.ndarray) -> bool:
        """
        Detect head turning based on normalized offset from shoulder center.
        """
        nose = kpts[self.NOSE]
        l_shoulder = kpts[self.LEFT_SHOULDER]
        r_shoulder = kpts[self.RIGHT_SHOULDER]

        # If either shoulder is missing, skip
        if np.allclose(l_shoulder, [0.0, 0.0]) or np.allclose(r_shoulder, [0.0, 0.0]):
            return False

        shoulder_center_x = (l_shoulder[0] + r_shoulder[0]) / 2
        shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
        if shoulder_width < 1:
            return False  # avoid division by zero

        horizontal_offset = abs(nose[0] - shoulder_center_x)
        normalized_offset = horizontal_offset / shoulder_width

        # Lower the threshold for better sensitivity
        return normalized_offset > 0.13


    def is_drowsy(self, kpts: np.ndarray) -> bool:
        """
        Detect head nodding down based on nose below shoulder line (in Y direction).
        """
        nose = kpts[self.NOSE]
        l_shoulder = kpts[self.LEFT_SHOULDER]
        r_shoulder = kpts[self.RIGHT_SHOULDER]
        shoulder_center_y = (l_shoulder[1] + r_shoulder[1]) / 2

        # If nose is significantly below shoulders -> possible drowsiness
        return (nose[1] - shoulder_center_y) > 40
