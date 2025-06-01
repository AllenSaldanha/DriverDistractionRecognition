import os
import cv2
import numpy as np
from src import YOLOv8Detector
from tqdm import tqdm

def extract_and_save_keypoints(video_path, output_folder, device=None):
    os.makedirs(output_folder, exist_ok=True)
    detector = YOLOv8Detector(device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in tqdm(range(total_frames), desc="Extracting keypoints"):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {frame_idx}")
            break

        results = detector.predict(frame)
        keypoints_list = detector.get_keypoints(results)

        # If no person detected, save empty array
        if keypoints_list is None or len(keypoints_list) == 0:
            keypoints_np = np.empty((0, 17, 2), dtype=np.float32)
        else:
            keypoints_np = keypoints_list.cpu().numpy()

        # Save keypoints for this frame
        out_file = os.path.join(output_folder, f"frame_{frame_idx:05d}.npy")
        np.save(out_file, keypoints_np)

    cap.release()
    print(f"Saved keypoints for {total_frames} frames to {output_folder}")

if __name__ == "__main__":
    video_path = "./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face.mp4"
    output_folder = "./keypoints/gA_1_s1"

    extract_and_save_keypoints(video_path, output_folder)
