import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
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

    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = I3D(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float('inf')
    writer = SummaryWriter(log_dir='runs/driver_activity_experiment')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (frame, labels) in enumerate(dataloader):
            # Shape: [batch, channels, frames, h, w]
            frame = frame.to(device)
            labels = labels.to(device)
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
        writer.add_scalar('Loss/train', epoch_loss, epoch)
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