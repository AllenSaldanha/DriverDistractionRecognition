from keypoint_dataset import DriverActivityKeypointDataset
from torch.utils.data import DataLoader
from utils.video_annotation_pairs import collect_video_annotation_pairs


pairs = collect_video_annotation_pairs('./dataset/dmd')
dataset = DriverActivityKeypointDataset(
    keypoints_folder='./keypoints/gA',
    video_annotation_pairs=pairs,
    num_frames=16  # Number of frames in each sequence (Like I3D)
)

# Wrap in DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Test loop
for keypoints, labels in dataloader:
    # keypoints.shape -> [B, num_of_frames, N, 17, 2] where N is person count 17 is COCO points
    # labels.shape -> [B, num_classes]
    print(keypoints.shape, labels.shape)
    break
