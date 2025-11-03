import torch 
import numpy as np
import torchvision
import torch.nn as nn
import glob
import os
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.optim as optim
from torchvision import transforms
import wandb    
import argparse
import joblib
from sklearn.decomposition import PCA
import timeit
from torch.nn import functional as F


from Decoder import UnetDecoder, AdapterUnetDecoder, VanillaUNet
from DataClasses import *
from OrganLabels import *
from losses import DiceCELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# directory of training data
data_dir = "Totalsegmentator_dataset_full"
uncertent_dir = "Uncertainties"


# vol_ds = TotalSegmentatorV2Dataset(
#     root_dir=data_dir,
#     group_map=GROUP_MAP,
# )

target_size = (256, 256)  # Change this as needed

# transforms for 2D slices
slice_transform = transforms.Compose([
transforms.Resize(target_size), # force all slices to same size
transforms.CenterCrop(target_size)
])

# transform for masks
mask_transform = transforms.Lambda(lambda mask: F.interpolate(
mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest'
).squeeze(0).squeeze(0).long())


# the three planes
planes = ["sagittal", "axial", "coronal"]
N = 10
# set seed
#np.random.seed(8)

# load in selected data for each plane
datasets = {}
for plane in planes:
    
    filepath = os.path.join(uncertent_dir, f"selected_subjects_{plane}.txt")
    with open(filepath, "r") as f:
        subjects = f.read().splitlines()
    if len(subjects)>N:
        subjects = subjects[:N]
    print(subjects)
    vol_ds = FilteredTotalSegmentatorV2Dataset(
    root_dir=data_dir,
    group_map=GROUP_MAP,
    subjects=subjects)

    data = CachedSliceDataset(
        base_dataset=vol_ds,
        plane=plane,
        slice_transform=slice_transform,
        mask_transform=mask_transform,
    )

    datasets[plane] = data



# combine datasets from all planes
combined_dataset = ConcatDataset([datasets[plane] for plane in planes])

print(f"Combined dataset size: {len(combined_dataset)}")

train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
print(f"Training set size: {train_size}")
print(f"Validation set size: {val_size}")
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# create data loader
batch_size = 32
train_loader =DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True,persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True,persistent_workers=True)

# build the model
sample_input = combined_dataset[0]
in_channels = sample_input["image"].shape[0]
print("Number of input channels:", in_channels)


num_classes = max(GROUP_MAP.values()) + 1

model = VanillaUNet(in_channels=in_channels, num_classes=num_classes)
model = model.to(device)

criterion = DiceCELoss().to(device)
lr = 1e-4
weight_decay = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

print(model)


modelpath = "VanillaUNet_model"
os.makedirs(modelpath, exist_ok=True)
modelpath = os.path.join(modelpath, "best_model.pth")


num_epochs = 200

name = "VanillaUNet_2D_fullDataset"
wandb.init(
    project="Master1",
    name=name,
    group="DINOv3_skip",
    config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "train_val_split": 0.8,
        "loss": "Dice + CrossEntropy",
        "optimizer": "Adam"
    }
)   


start_time = timeit.default_timer()
print("Starting training...")
val_every = 5  # validate every n epochs
best_val_loss = float('inf')

# lists for tracking per-epoch losses
train_losses = []
train_ce_losses = []
train_dice_losses = []
val_losses = []

for epoch in range(num_epochs):
    # ---------------- TRAIN ----------------
    model.train()
    ce_running = 0.0
    dice_running = 0.0
    total_loss = 0.0
    total_batches = 0

    for batch in train_loader:
        inputs = batch["image"].to(device)
        labels = batch["mask"].to(device).squeeze(1)  # [B, H, W]

        outputs = model(inputs)
        loss, ce_loss, dice_loss = criterion(outputs, labels)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        ce_running += ce_loss
        dice_running += dice_loss
        total_batches += 1

        print(f"processed batch {total_batches} in epoch {epoch+1}", end='\r')

        #print("Updated model parameters")


    
    # compute epoch averages
    avg_loss = float(total_loss / total_batches)
    avg_ce = float(ce_running / total_batches)
    avg_dice = float(dice_running / total_batches)

    train_losses.append(avg_loss)
    train_ce_losses.append(avg_ce)
    train_dice_losses.append(avg_dice)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Dice: {avg_dice:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train/loss": avg_loss,
        "train/ce_loss": avg_ce,
        "train/dice_loss": avg_dice
    })

    # ---------------- VALIDATION ----------------
    if (epoch + 1) % val_every == 0:
        model.eval()
        ce_val_total = 0.0
        dice_val_total = 0.0
        loss_val_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["image"].to(device)
                labels = batch["mask"].to(device).squeeze(1)  

                outputs = model(inputs)

                loss, ce_loss, dice_loss = criterion(outputs, labels)

                loss_val_total += loss
                ce_val_total += ce_loss
                dice_val_total += dice_loss
                val_batches += 1
     

        avg_val_loss = float(loss_val_total / val_batches)
        avg_val_ce = float(ce_val_total / val_batches)
        avg_val_dice = float(dice_val_total / val_batches)

        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Validation Loss: {avg_val_loss:.4f}, CE: {avg_val_ce:.4f}, Dice: {avg_val_dice:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "val/loss": avg_val_loss,
            "val/ce_loss": avg_val_ce,
            "val/dice_loss": avg_val_dice
        })

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), modelpath)
            print(f"Best model saved to path: {modelpath}")

            wandb.log({"best_val_loss": best_val_loss})
            wandb.save(modelpath)

end_time = timeit.default_timer()
elapsed = end_time - start_time
print(f"Total training time: {elapsed/60:.2f} minutes")
print("Training complete.")
wandb.finish()

# Save all loss histories
np.save(os.path.join(os.path.dirname(modelpath), "train_losses.npy"), np.array(train_losses))
np.save(os.path.join(os.path.dirname(modelpath), "train_ce_losses.npy"), np.array(train_ce_losses))
np.save(os.path.join(os.path.dirname(modelpath), "train_dice_losses.npy"), np.array(train_dice_losses))
np.save(os.path.join(os.path.dirname(modelpath), "val_losses.npy"), np.array(val_losses))




    
