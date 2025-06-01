import torch
import cv2
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # Load model from TorchHub
        self.model = YOLO('yolov8n-pose.pt')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, frame):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Run inference
        results = self.model(img_rgb)
        return results

    def draw_results(self, frame, results):
        result = results[0]
        annotated_img = result.plot()
        # Convert back to BGR for OpenCV display
        frame = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        return frame
    
    def get_keypoints(self, results):
        # Extract keypoints from the first result
        keypoints = results[0].keypoints
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return None
        # keypoints.xy is a tensor of shape (N, 17, 2) for N detected persons
        return keypoints.xy  # Returns list-like tensor of (17, 2) arrays for each person

