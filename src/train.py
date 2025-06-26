import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import DriverActivityDataset
from models.I3D import I3D

NUM_CLASSES = 12
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_FRAMES = 32

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

dataset = DriverActivityDataset(
    video_path='./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face.mp4',
    annotation_json_path='./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json',
    transform=transform,
    clip_len=16
)

dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = I3D(num_classes=NUM_CLASSES).to(device)


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (frame, labels) in enumerate(dataloader):
        # Shape: [batch, channels, frames, h, w]
        frame = frame.to(device)
        labels = labels.to(device)
        break
    break
