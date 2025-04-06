import torch
from utils.metrics import iou_score, dice_score

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            pred_mask, _ = model(images)

            loss = criterion(pred_mask, masks)
            val_loss += loss.item()

            val_iou += iou_score(torch.sigmoid(pred_mask), masks).item()
            val_dice += dice_score(torch.sigmoid(pred_mask), masks).item()

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)

    return val_loss, val_iou, val_dice
