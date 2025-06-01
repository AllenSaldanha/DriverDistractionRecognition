import json
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class DriverActivityDataset(Dataset):
    def __init__(self, video_path, annotation_json_path, transform=None):
        self.video_path = video_path
        self.transform = transform if transform is not None else T.ToTensor()

        # Load OpenLabel JSON
        with open(annotation_json_path, 'r') as f:
            data = json.load(f)
        self.openlabel_data = data["openlabel"]

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
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Could not read frame {idx} from {self.video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        if self.transform:
            pil_image = self.transform(pil_image)
        action_labels = self._extract_info_for_frame(idx, self.actions_data)

        return pil_image, action_labels