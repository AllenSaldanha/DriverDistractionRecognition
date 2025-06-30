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
    def __init__(self, keypoints_folder, video_annotation_pairs, num_frames=16):
        self.keypoints_folder = Path(keypoints_folder)
        self.action_classes = load_trained_classes("./src/trained_classes.txt")
        self.num_classes = len(self.action_classes)
        self.action_to_idx = {name: i for i, name in enumerate(self.action_classes)}
        self.num_frames = num_frames
        
        self.video_meta = []
        self.samples = []
        
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
            
            if total_frames < self.num_frames:
                print(f"Warning: Video {video_path_stem} has only {total_frames} frames, skipping...")
                continue
            
            self.video_meta.append((keypoints_dir, ann_path, total_frames, actions))
            
            # Just like in I3D, we will sample sequences of frames
            for start_frame in range(0, total_frames - self.num_frames + 1, self.num_frames):
                self.samples.append((keypoints_dir, ann_path, start_frame, actions))
            
               
    def __len__(self):
        return len(self.samples)

    def _extract_labels_for_frame(self, frame_idx, actions_data):
        """Extract multi-hot labels for a specific frame"""
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        
        for _ , info in actions_data.items():
            label = info.get("type", "Unknown")
            if label not in self.action_to_idx:
                continue  # Skip unknown labels
                
            for interval in info.get("frame_intervals", []):
                if interval["frame_start"] <= frame_idx <= interval["frame_end"]:
                    label_vec[self.action_to_idx[label]] = 1.0
                    break
                    
        return label_vec

    def __getitem__(self, idx):
        keypoints_dir, _ , start_frame, actions_data = self.samples[idx]
        
        sequence_keypoints = []
        sequence_labels = []
        
        for i in range(self.num_frames):
            frame_idx = start_frame + i
            keypoint_file = keypoints_dir / f"frame_{frame_idx:05d}.npy"
            
            try:
                keypoints_np = np.load(str(keypoint_file))
            except FileNotFoundError:
                print(f"Warning: Keypoint file {keypoint_file} not found, using zeros")
                keypoints_np = np.empty((0, 17, 2), dtype=np.float32)
            
            if keypoints_np.shape[0] == 0:
                # No person detected, use zeros
                keypoints_np = np.zeros((1, 17, 2), dtype=np.float32)
            elif keypoints_np.shape[0] > 1:
                # More than one person detected, use only the first
                keypoints_np = keypoints_np[:1]
            
            # Extract labels for this frame
            frame_labels = self._extract_labels_for_frame(frame_idx, actions_data)
            
            sequence_keypoints.append(keypoints_np)
            sequence_labels.append(frame_labels)
        
        # Convert to tensors
        # Shape: [num_frames, person, 17, 2]
        keypoints_tensor = torch.tensor(np.stack(sequence_keypoints), dtype=torch.float32)
        
        # Shape: [num_classes]
        labels_tensor = torch.stack(sequence_labels).max(dim=0).values
        
        return keypoints_tensor, labels_tensor
