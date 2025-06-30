import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src import YOLOv8Detector
from src.utils.video_annotation_pairs import collect_video_annotation_pairs

def extract_and_save_keypoints(root_dir, output_folder, device=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    detector = YOLOv8Detector(device=device)
    
    video_paths = [x[0] for x in collect_video_annotation_pairs(root_dir)]

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_dir = output_folder / Path(video_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in tqdm(range(total_frames), desc=f"Extracting keypoints from {video_path}"):
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
            out_file = out_dir / f"frame_{frame_idx:05d}.npy"
            np.save(str(out_file), keypoints_np)

        cap.release()
        print(f"Saved keypoints for {total_frames} frames to {out_dir}")

if __name__ == "__main__":
    root_dir = "./dataset/dmd/gA"
    output_folder = "./keypoints/gA"
    extract_and_save_keypoints(root_dir, output_folder)
