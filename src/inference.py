import os
import argparse
import torch
import csv
import logging

from pathlib import Path
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import DriverActivityDataset, load_trained_classes
from utils.video_annotation_pairs import collect_video_annotation_pairs
from models.I3D import I3D

logging.basicConfig(filename='inference.log', level=logging.INFO)

def inference(pair, model_path):
    # This function should load the model and perform inference on the video clips
    
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
    trained_classes = load_trained_classes("./src/trained_classes.txt")
    model = I3D(num_classes=len(trained_classes))
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    threshold = 0.5
    
    video_to_frames_preds = {}
    
    for i, (video_tensor, label_tensor) in enumerate(dataloader):
        video_tensor = video_tensor.to(device)
        label_tensor = label_tensor.to(device)

        with torch.no_grad():
            outputs = model(video_tensor)
            predictions = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
            # since multi-class classification, use a threshold to determine the predicted classes
            pred_labels = [trained_classes[j] for j, p in enumerate(predictions) if p >= threshold]
            
            video_path, _, start_frame = dataset.samples[i]
            video_name = Path(video_path).stem
            row = [start_frame, ";".join(pred_labels)]
            
            if video_name not in video_to_frames_preds:
                video_to_frames_preds[video_name] = []
            video_to_frames_preds[video_name].append(row)
            
    # Save predictions to .csv files
    for video_name, frames_preds in video_to_frames_preds.items(): 
        output_file = os.path.join("inference_logs/", f"{video_name}.csv")
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame Start', 'Predicted Labels'])
            writer.writerows(frames_preds)
        logging.info(f"Saved predictions for {video_name} to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Driver Activity Inference")
    parser.add_argument("--root_dir", default = "./dataset/dmd/gA/2", type=str, help="Path to dataset root")
    parser.add_argument("--model_path", default="final_model.pth", type=str, help="Path to trained model (.pth)")
    
    args = parser.parse_args()

    pairs = collect_video_annotation_pairs(args.root_dir)
    print(f"Found {len(pairs)} valid video-annotation pairs.")

    inference(pairs, model_path=args.model_path)