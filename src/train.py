import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import logging
import argparse
from dataset import DriverActivityDataset
from models.I3D import I3D
from utils.video_annotation_pairs import collect_video_annotation_pairs

logging.basicConfig(filename='training.log', level=logging.INFO)

def main(pairs):
    EPOCHS = 6

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    dataset = DriverActivityDataset(
        video_annotation_pairs=pairs,
        transform=transform,
        num_frames=16
    )

    # Split dataset into training and validation sets (80%-20%)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=30, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=30, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model initialization
    model = I3D(num_classes=dataset.num_classes)
    model = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    best_loss = float('inf')
    writer = SummaryWriter(log_dir='runs/driver_activity_experiment')

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, (frame, labels) in enumerate(train_loader):
            frame = frame.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(frame)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}] Training Loss: {epoch_loss:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)

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
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"Saved best model at epoch {epoch+1} with val loss {val_loss:.4f}")

        scheduler.step()
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        writer.flush()

    # Save final model checkpoint
    torch.save(model.state_dict(), 'final_model.pth')
    logging.info("Final model saved.")
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Driver Activity I3D Training")
    parser.add_argument("--root_dir", default = "./dataset/dmd", type=str, help="Path to dataset root")
    args = parser.parse_args()

    pairs = collect_video_annotation_pairs(args.root_dir)
    print(f"Found {len(pairs)} valid video-annotation pairs.")

    main(pairs)
