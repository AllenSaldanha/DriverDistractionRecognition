import os
import json
import numpy as np
import torch
from pathlib import Path
import glob

from torch.utils.data import Dataset

def load_trained_classes(path):
    """Load action classes from respective files"""
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

class DriverActivityKeypointDataset(Dataset):
    def __init__(self, keypoints_folder, video_annotation_pairs):
        self.keypoints_folder = Path(keypoints_folder)
        self.action_classes = load_trained_classes("./src/trained_classes.txt")
        self.num_classes = len(self.action_classes)
        self.action_to_idx = {name: i for i, name in enumerate(self.action_classes)}
        
        self.video_meta = []
        
        # Iterate through video_annotation_pairs to load annotations and their respective keypoints
        for video_path, ann_path in video_annotation_pairs:
            video_path_stem = Path(video_path).stem
            keypoints_dir = self.keypoints_folder / video_path_stem

            if not keypoints_dir.exists():
                print(f"Warning: Keypoints directory {keypoints_dir} not found, skipping...")
                continue

            with open(ann_path, 'r') as f:
                data = json.load(f)["openlabel"]

            actions = data.get("actions", {})

            # List .npy keypoint files sorted by frame index
            keypoint_files = sorted(glob.glob(str(keypoints_dir / "frame_*.npy")))
            total_frames = len(keypoint_files)
            
            self.video_meta.append((keypoints_dir, ann_path, total_frames, actions))
            print(self.video_meta)
            break
               
    def __len__(self):
        return len(self.keypoint_files)

    def _extract_labels_for_frame(self, frame_number):
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for obj_id, obj_info in self.actions_data.items():
            label = obj_info.get("type", "Unknown")
            intervals = obj_info.get("frame_intervals", [])
            for interval in intervals:
                if interval["frame_start"] <= frame_number <= interval["frame_end"]:
                    if label in self.action_to_idx:
                        labels[self.action_to_idx[label]] = 1.0
                    break
        return labels

    def __getitem__(self, idx):
        keypoint_path = os.path.join(self.keypoints_folder, self.keypoint_files[idx])
        keypoints_np = np.load(keypoint_path)  # shape: (N, 17, 2)

        # If no persons detected, return zeros
        if keypoints_np.size == 0:
            keypoints_tensor = torch.zeros((1, 17, 2), dtype=torch.float32)
        else:
            keypoints_tensor = torch.tensor(keypoints_np, dtype=torch.float32)

        action_labels = self._extract_labels_for_frame(idx)  # Multi-hot label
        return keypoints_tensor, action_labels
