import json
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class DriverActivityDataset(Dataset):
    def __init__(self, video_path, annotation_json_path, num_frames, transform=None):
        self.video_path = video_path
        self.transform = transform if transform is not None else T.ToTensor()
        self.num_frames = num_frames
        
        # Load OpenLabel JSON
        with open(annotation_json_path, 'r') as f:
            data = json.load(f)
        try:
            self.openlabel_data = data["openlabel"]
        except KeyError:
            raise ValueError(f"Missing 'openlabel' key in {annotation_json_path}")

        # Get total number of frames from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Preprocess object/action annotations
        self.objects_data = self.openlabel_data.get("objects", {})
        self.actions_data = self.openlabel_data.get("actions", {})

        # Define all possible action labels
        self.action_classes = sorted(list(set(
            obj_info.get("type", "Unknown") for obj_id, obj_info in self.actions_data.items()
        )))
        self.num_classes = len(self.action_classes)
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_classes)}

    def __len__(self):
        return self.total_frames

    def _extract_info_for_frame(self, frame_number, object_dict):
        """Return multi-hot encoded tensor for actions active at the frame."""
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for obj_id, obj_info in object_dict.items():
            label = obj_info.get("type", "Unknown")
            intervals = obj_info.get("frame_intervals", [])
            for interval in intervals:
                if interval["frame_start"] <= frame_number <= interval["frame_end"]:
                    if label in self.action_to_idx:
                        labels[self.action_to_idx[label]] = 1.0
                    break  # Only need to match one interval
        return labels

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adjust starting index so we don’t go out of bounds
        if idx + self.num_frames > total_frames:
            idx = max(0, total_frames - self.num_frames)

        frames = []
        labels = []

        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx + i)
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[Warning] Could not read frame {idx + i} from {self.video_path}. Returning dummy.")
                dummy = torch.zeros((3, 224, 224))
                frames.append(dummy)
                labels.append(torch.zeros(self.num_classes))
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)

            if self.transform:
                pil_image = self.transform(pil_image)

            frames.append(pil_image)

            label = self._extract_info_for_frame(idx + i, self.actions_data)
            labels.append(label)

        cap.release()

        # Shape: [clip_len, C, H, W] → [C, clip_len, H, W]
        frames_tensor = torch.stack(frames).permute(1, 0, 2, 3)
        label_tensor = torch.stack(labels).max(dim=0).values

        return frames_tensor, label_tensor
