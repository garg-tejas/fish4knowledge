import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from datasets.fish_dataset import FishDataset
from models.multitask_model import FishModel
from utils.evaluate import validate

from tqdm import tqdm

# ==== CONFIGURATION ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
batch_size = 16
lr = 1e-4
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

writer = SummaryWriter(log_dir="runs/fish_experiment")

# ==== DATASETS ====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = FishDataset(root_dir="data/train", transform=transform)
val_dataset = FishDataset(root_dir="data/val", transform=transform)
test_dataset = FishDataset(root_dir="data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ==== MODEL ====
model = FishModel(num_classes=23).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_mask = nn.BCEWithLogitsLoss()
criterion_class = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler('cuda')

best_val_loss = float('inf')

# ==== TRAINING LOOP ====
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for images, masks, labels in loop:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            pred_mask, pred_class = model(images)

            loss_mask = criterion_mask(pred_mask, masks)
            loss_class = criterion_class(pred_class, labels)

            loss = loss_mask + loss_class

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f}")

    writer.add_scalar('Loss/train', epoch_loss, epoch)

    # ==== VALIDATION ====
    val_loss, val_iou, val_dice = validate(model, val_loader, device)

    print(f"Epoch [{epoch+1}/{epochs}] Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} Dice: {val_dice:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{save_dir}/best_model_epoch{epoch+1}.pth")
        print("✅ Best model saved!")

    torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch+1}.pth")

writer.close()

# ==== FINAL TEST EVALUATION ====
print("\nEvaluating on test set...")
test_loss, test_iou, test_dice = validate(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f} IoU: {test_iou:.4f} Dice: {test_dice:.4f}")