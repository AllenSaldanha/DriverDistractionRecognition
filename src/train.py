import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import logging
from dataset import DriverActivityDataset
from models.I3D import I3D

logging.basicConfig(filename='training.log', level=logging.INFO)

def main():
    NUM_CLASSES = 12
    EPOCHS = 10

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    dataset = DriverActivityDataset(
        video_path='./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face.mp4',
        annotation_json_path='./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json',
        transform=transform,
        num_frames=16
    )

    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=5, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model initialization
    model = I3D(num_classes=NUM_CLASSES)
    model = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    best_loss = float('inf')
    writer = SummaryWriter(log_dir='runs/driver_activity_experiment')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (frame, labels) in enumerate(dataloader):
            frame = frame.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Training steps
            optimizer.zero_grad()
            outputs = model(frame)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}] Loss: {epoch_loss:.4f}")
        scheduler.step()
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        writer.flush()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"Saved best model at epoch {epoch+1} with loss {epoch_loss:.4f}")
        
    torch.save(model.state_dict(), 'i3d_driver_activity_rgb.pth')
    logging.info("Model saved.")
    writer.close()
    
if __name__ == '__main__':
    main()