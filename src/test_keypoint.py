from keypoint_dataset import DriverActivityKeypointDataset
from torch.utils.data import DataLoader

dataset = DriverActivityKeypointDataset(
    keypoints_folder='./keypoints/gA_1_s1',
    annotation_json_path='./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json'
)

# Wrap in DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Test loop
for keypoints, labels in dataloader:
    # keypoints.shape -> [B, N, 17, 2] where N is person count 17 is COCO points
    # labels.shape -> [B, num_classes]
    print(keypoints, labels)
    break
