import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class DriverActivityKeypointDataset(Dataset):
    def __init__(self, keypoints_folder, annotation_json_path):
        self.keypoints_folder = keypoints_folder

        with open(annotation_json_path, 'r') as f:
            data = json.load(f)
        self.openlabel_data = data["openlabel"]

        self.actions_data = self.openlabel_data.get("actions", {})

        # Define action classes
        self.action_classes = sorted(list(set(
            obj_info.get("type", "Unknown") for obj_id, obj_info in self.actions_data.items()
        )))
        self.num_classes = len(self.action_classes)
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_classes)}

        # List .npy keypoint files sorted by frame index
        self.keypoint_files = sorted([
            f for f in os.listdir(keypoints_folder) if f.endswith('.npy')
        ])
    
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
