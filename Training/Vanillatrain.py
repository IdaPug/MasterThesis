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
import timeit
from torch.nn import functional as F
from torch.utils.data import Dataset
import h5py
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import argparse
import random

from Decoder import UnetDecoder, AdapterUnetDecoder, VanillaUNet

from DataClasses import *
from OrganLabels import *
from losses import DiceCELoss

    

def main ():

    parser = argparse.ArgumentParser(description="Train Vanilla UNet on Slices with varying training set sizes")
    parser.add_argument('--TrainSizePercent', type=float, default=0.1, help='Percentage of subjects to use for training of of total number (0-1)')
    parser.add_argument('--Seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--Epochs', type=int, default=200, help='Number of training epochs')

    args = parser.parse_args()

    # check that train size percent is valid
    if args.TrainSizePercent <= 0 or args.TrainSizePercent > 100:
        raise ValueError("TrainSizePercent must be between 0 and 100")

    if args.Seed is not None:
        torch.manual_seed(args.Seed)
        np.random.seed(args.Seed)
        random.seed(args.Seed)
        print(f"Set seed to {args.Seed}")
    else:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        print("No seed set. Using default seed 42.")

    if wandb.run is not None:  # inside a sweep
        args.TrainSizePercent = wandb.config.TrainSizePercent
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # directory of training data
    data_dir = "../TotalSegmentator_v2_Full/TrainSubjects/"

    available_subjects = [d for d in glob.glob(os.path.join(data_dir, "s*")) if os.path.isdir(d)]
    subjects_names = [os.path.basename(d) for d in available_subjects]
    print(f"Total available subjects in data dir: {len(available_subjects)}")
    
    planes = ["sagittal", "axial", "coronal"]
    train_dataset = {}
    val_dataset = {}
    for plane in planes:
        ds = Slicedataset(data_dir, plane, subjects_names)
        # take percent of dataset randomly
        num_subjects = len(ds)
        num_train = int((args.TrainSizePercent / 100) * num_subjects)
        num_val = int((0.05) * num_subjects)  # 5% for validation
        print()

        random_indices = random.sample(range(num_subjects), num_train)
        val_indices = random.sample(range(num_subjects), num_val)

        train_subset = torch.utils.data.Subset(ds, random_indices)
        val_subset = torch.utils.data.Subset(ds, val_indices)

        train_dataset[plane] = train_subset
        val_dataset[plane] = val_subset


 

    # combine datasets from different planes
    train_dataset = ConcatDataset(list(train_dataset.values()))
    val_dataset = ConcatDataset(list(val_dataset.values()))
    print(f"Training subjects: {len(train_dataset)} slices")
    print(f"Validation subjects: {len(val_dataset)} slices")

    
    # create data loader
    batch_size = 32
    train_loader =DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    # build the model
    sample_input = train_dataset[0]
    print("Sample input shape:", sample_input["image"].shape)
    in_channels = sample_input["image"].shape[1]
    print("Number of input channels:", in_channels)


    num_classes = max(GROUP_MAP.values()) + 1

    model = VanillaUNet(in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)
    
    criterion = DiceCELoss().to(device)
    lr = 1e-4
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    



    mainpath = "VanillaUNet_model_BetterData"
    subpath  = f"TrainSize{args.TrainSizePercent}/"
    modelpath = os.path.join(mainpath, subpath)
    os.makedirs(modelpath, exist_ok=True)

    modelpath = os.path.join(modelpath, "best_model.pth")


    num_epochs = args.Epochs

    name = "VanillaUNet_2D_fullDataset_TrainSize_" + str(args.TrainSizePercent)
    wandb.init(
        project="VanillaUNet_2D",
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
        # TRAIN 
        model.train()
        ce_running = 0.0
        dice_running = 0.0
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            inputs = batch["image"].to(device).squeeze(1)  # [B, C, H, W]
            labels = batch["mask"].to(device).squeeze(1)  # [B, H, W]

            torch.cuda.reset_peak_memory_stats()
            outputs = model(inputs)
            loss, ce_loss, dice_loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory: {peak_mem:.2f} GB")
            
            total_loss += loss
            ce_running += ce_loss
            dice_running += dice_loss
            total_batches += 1
        
            #print(f"processed batch {total_batches} in epoch {epoch+1}", end='\r')

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
                    inputs = batch["image"].to(device).squeeze(1)
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


if __name__ == "__main__":
    main()



    
