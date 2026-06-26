"""
Training script for SegFormer semantic segmentation.

This script:
- Loads the training and validation datasets.
- Fine-tunes the SegFormer model.
- Evaluates the model using Pixel Accuracy, IoU and Dice Score.
- Saves the best-performing model based on validation IoU.

Author: Abhimanyu Upadhyay
"""
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

from config import *
from dataset import TurfDataset
from metrics import evaluate_metrics
torch.manual_seed(SEED)

# -----------------------
# Dataset
# -----------------------
train_dataset = TurfDataset(TRAIN_IMAGES, TRAIN_MASKS)
val_dataset = TurfDataset(VAL_IMAGES, VAL_MASKS)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# -----------------------
# Model
# -----------------------
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

model.to(DEVICE)

# -----------------------
# Optimizer
# -----------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

best_iou = 0.0

# -----------------------
# Training Loop
# -----------------------
for epoch in range(NUM_EPOCHS):

    model.train()

    train_loss = 0

    progress = tqdm(train_loader)

    for batch in progress:

        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        progress.set_description(
            f"Epoch {epoch+1}/{NUM_EPOCHS} Loss {loss.item():.4f}"
        )

    # -----------------------
    # Validation
    # -----------------------
    model.eval()

    total_iou = 0
    total_dice = 0
    total_acc = 0

    with torch.no_grad():

        for batch in val_loader:

            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values)

            metrics = evaluate_metrics(
                outputs.logits,
                labels
            )

            total_acc += metrics["pixel_accuracy"]
            total_iou += metrics["iou"]
            total_dice += metrics["dice"]

    total_acc /= len(val_loader)
    total_iou /= len(val_loader)
    total_dice /= len(val_loader)

    print("\n")
    print(f"Epoch {epoch+1}")
    print(f"Train Loss : {train_loss/len(train_loader):.4f}")
    print(f"Pixel Acc  : {total_acc:.4f}")
    print(f"IoU        : {total_iou:.4f}")
    print(f"Dice       : {total_dice:.4f}")

    if total_iou > best_iou:

        best_iou = total_iou

        torch.save(
            model.state_dict(),
            WEIGHTS_DIR / "best_model.pth"
        )

        print("=" * 50)
        print(f"Best model saved! (IoU: {best_iou:.4f})")
        print("=" * 50)

print("\n" + "=" * 60)
print("Training completed successfully.")
print(f"Best Validation IoU : {best_iou:.4f}")
print(f"Model saved to      : {WEIGHTS_DIR / 'best_model.pth'}")
print("=" * 60)