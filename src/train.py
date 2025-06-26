import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import DriverActivityDataset
from models.I3D import I3D

NUM_CLASSES = 12

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

dataset = DriverActivityDataset(
    video_path='./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face.mp4',
    annotation_json_path='./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = I3D(num_classes=NUM_CLASSES).to(device)

for frame, labels in dataloader:
    # training step here
    # print(frame, labels)
    pass
