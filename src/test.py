from dataset import DriverActivityDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

dataset = DriverActivityDataset(
    video_path='./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face.mp4',
    annotation_json_path='./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json',
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for frame, labels in dataloader:
    # training step here
    print(frame, labels)
