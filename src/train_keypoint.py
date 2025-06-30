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

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for i, (keypoints, labels) in enumerate(train_loader):
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
    
    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for keypoints, labels in val_loader:
            keypoints = keypoints.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

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
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logging.info(f"Training Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # Validation
        val_loss = validate_epoch(model, val_loader, criterion, device)
        logging.info(f"Validation Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning Rate: {current_lr:.6f}")
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f'best_keypoint_{args.model_type}_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_type': args.model_type,
                'num_classes': dataset.num_classes,
                'action_classes': dataset.action_classes,
                'args': args
            }, best_model_path)
            logging.info(f"Saved best model at epoch {epoch+1} with val loss {val_loss:.4f}")
        
        
        writer.flush()
        logging.info("")
    
    # Save final model
    final_model_path = f'final_keypoint_{args.model_type}_model.pth'
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_type': args.model_type,
        'num_classes': dataset.num_classes,
        'action_classes': dataset.action_classes,
        'args': args
    }, final_model_path)
    
    logging.info(f"Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Final model saved to: {final_model_path}")
    
    writer.close()
        
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
    