import json
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os

def load_trained_classes(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

class DriverActivityDataset(Dataset):
    def __init__(self, video_annotation_pairs, num_frames=16, transform=None):
        self.pairs = video_annotation_pairs
        self.transform = transform if transform is not None else T.ToTensor()
        self.num_frames = num_frames

        self.samples = []
        self.action_classes = load_trained_classes("./src/trained_classes.txt")
        
        self.video_meta = []  # [(video_path, ann_path, total_frames, actions_data)]
        for video_path, ann_path in self.pairs:
            with open(ann_path, 'r') as f:
                try:
                    data = json.load(f)["openlabel"]
                except KeyError:
                    raise ValueError(f"Missing 'openlabel' key in {ann_path}")

            actions = data.get("actions", {})
            total_frames = self._get_total_frames(video_path)
            self.video_meta.append((video_path, ann_path, total_frames, actions))
            self.num_classes = len(self.action_classes)
            self.action_to_idx = {name: i for i, name in enumerate(self.action_classes)}

        for video_path, ann_path, total_frames, _ in self.video_meta:
            for start_frame in range(0, total_frames - self.num_frames + 1, self.num_frames):
                self.samples.append((video_path, ann_path, start_frame))

    def _get_total_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total

    def __len__(self):
        return len(self.samples)

    def _extract_labels_for_frame(self, frame_idx, actions_data):
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for action_id, info in actions_data.items():
            label = info.get("type", "Unknown")
            if label not in self.action_to_idx:
                continue  # skip unknown labels
            for interval in info.get("frame_intervals", []):
                if interval["frame_start"] <= frame_idx <= interval["frame_end"]:
                    label_vec[self.action_to_idx[label]] = 1.0
                    break
        return label_vec

    def __getitem__(self, idx):
        video_path, ann_path, start_frame = self.samples[idx]

        # Load annotations
        with open(ann_path, 'r') as f:
            actions_data = json.load(f)["openlabel"].get("actions", {})

        # Open video
        cap = cv2.VideoCapture(video_path)
        frames = []
        labels = []

        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[Warning] Could not read frame {start_frame + i} from {video_path}")
                dummy = torch.zeros((3, 224, 224))
                frames.append(dummy)
                labels.append(torch.zeros(self.num_classes))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            if self.transform:
                image = self.transform(image)

            frames.append(image)
            labels.append(self._extract_labels_for_frame(start_frame + i, actions_data))

        cap.release()

        video_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]
        label_tensor = torch.stack(labels).max(dim=0).values    # Multi-hot vector

        return video_tensor, label_tensor