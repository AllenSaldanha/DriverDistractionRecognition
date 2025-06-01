import cv2, os

class FrameLoader:
    # iterator class for frame loading
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

    def __iter__(self):
        return self
    
    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame
    
    def release(self):
        self.cap.release()
