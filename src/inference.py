import os
import argparse
import torch

import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import DriverActivityDataset
from utils.video_annotation_pairs import collect_video_annotation_pairs
from models.I3D import I3D

def inference(pair, model_path, output_dir):
    # This function should load the model and perform inference on the video frames
    os.makedirs(output_dir, exist_ok=True)
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    
    dataset = DriverActivityDataset(
        video_annotation_pairs=pair,
        num_frames=16,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO 
    # Current model has 21 classes, but it was an error
    # Actual classes 22, so change num_classes to dataset.num_classes with the new saved model
    model = I3D(num_classes=21)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    for video_tensor, label_tensor in dataloader:
        video_tensor = video_tensor.to(device)
        label_tensor = label_tensor.to(device)

        print(video_tensor.shape, label_tensor.shape)
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Driver Activity Inference")
    parser.add_argument("--root_dir", default = "./dataset/dmd/gA/2", type=str, help="Path to dataset root")
    parser.add_argument("--model_path", default="final_model.pth", type=str, help="Path to trained model (.pt)")
    parser.add_argument("--output_dir", default="./inference_logs", type=str, help="Where to save .log predictions")
    
    args = parser.parse_args()

    pairs = collect_video_annotation_pairs(args.root_dir)
    print(f"Found {len(pairs)} valid video-annotation pairs.")

    inference(pairs, model_path=args.model_path, output_dir=args.output_dir)