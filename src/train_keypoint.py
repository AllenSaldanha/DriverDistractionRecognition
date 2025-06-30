import torch
import argparse
import logging

from torch.utils.data import DataLoader, random_split
from keypoint_dataset import DriverActivityKeypointDataset
from utils.video_annotation_pairs import collect_video_annotation_pairs

logging.basicConfig(filename='training.log', level=logging.INFO)

def main(pairs, keypoints_folder):
    EPOCHS = 6
    
    dataset = DriverActivityKeypointDataset(
        keypoints_folder= keypoints_folder,
        video_annotation_pairs=pairs,
        num_frames=16  # Number of frames in each sequence (Like I3D)
    )

     # Split dataset into training and validation sets (80%-20%)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=30, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=30, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test loop
    for keypoints, labels in train_loader:
        # keypoints.shape -> [B, num_of_frames, N, 17, 2] where N is person count 17 is COCO points
        # labels.shape -> [B, num_classes]
        print(keypoints.shape, labels.shape)
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Driver Activity Keypoint Training")
    parser.add_argument("--root_dir", default = "./dataset/dmd", type=str, help="Path to dataset root")
    parser.add_argument("--keypoints_folder", default="./keypoints/gA", type=str, help="Path to keypoints folder")
    args = parser.parse_args()

    pairs = collect_video_annotation_pairs(args.root_dir)
    keypoints_folder = args.keypoints_folder
    print(f"Found {len(pairs)} valid video-annotation pairs.")
    
    main(pairs, keypoints_folder)
    