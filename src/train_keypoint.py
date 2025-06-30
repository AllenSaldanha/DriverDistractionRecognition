import torch
import argparse
import logging
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from keypoint_dataset import DriverActivityKeypointDataset
from models.LSTM import (KeypointLSTM, KeypointGRU, KeypointTransformer, 
                           KeypointCNN1D, KeypointAttentionLSTM)
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

    # Model selection
    model_dict = {
        'lstm': KeypointLSTM,
        'gru': KeypointGRU,
        'transformer': KeypointTransformer,
        'cnn1d': KeypointCNN1D,
        'attention_lstm': KeypointAttentionLSTM
    }
    
    # Initialize model
    model_class = model_dict[args.model_type]
    
    if args.model_type == 'transformer':
        model = model_class(
            num_classes=dataset.num_classes,
            max_persons=1,
            d_model=args.hidden_size,
            nhead=8,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        model = model_class(
            num_classes=dataset.num_classes,
            max_persons=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    
    model = model.to(device)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model: {args.model_type}")
    logging.info(f"Number of parameters: {num_params:,}")
    logging.info(f"Model device: {next(model.parameters()).device}")
    
    
     # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    # TensorBoard logging
    log_dir = f'runs/keypoint_{args.model_type}_experiment'
    writer = SummaryWriter(log_dir=log_dir)
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        logging.info("-" * 50)
        
        model.train()
        running_loss = 0.0
        
        for i, (keypoints, labels) in enumerate(train_loader):
            # keypoints.shape -> [B, num_of_frames, N, 17, 2] where N is person count 17 is COCO points
            # labels.shape -> [B, num_classes]
            keypoints = keypoints.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                logging.info(f"Epoch [{epoch+1}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
       
        epoch_loss = running_loss / len(train_loader)
        logging.info(f"Training Loss: {epoch_loss:.4f}")
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frame, labels in val_loader:
                frame = frame.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(frame)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logging.info(f"Epoch [{epoch+1}] Validation Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"Saved best model at epoch {epoch+1} with val loss {val_loss:.4f}")

        scheduler.step()
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        writer.flush()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Driver Activity Keypoint Training")
    parser.add_argument("--root_dir", default = "./dataset/dmd", type=str, help="Path to dataset root")
    parser.add_argument("--keypoints_folder", default="./keypoints/gA", type=str, help="Path to keypoints folder")
    
    
    # Model params
    parser.add_argument("--model_type", type=str, default="lstm", 
                       choices=['lstm', 'gru', 'transformer', 'cnn1d', 'attention_lstm'],
                       help="Type of model to train")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden size for RNN/Transformer models")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    
    args = parser.parse_args()

    pairs = collect_video_annotation_pairs(args.root_dir)
    keypoints_folder = args.keypoints_folder
    print(f"Found {len(pairs)} valid video-annotation pairs.")
    
    main(pairs, keypoints_folder)
    