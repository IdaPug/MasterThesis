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
import sys
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torch import amp
scaler = amp.GradScaler('cuda')

from Decoder import UnetDecoder, AdapterUnetDecoder, VanillaUNet

from ArgumentaitonClasses import GaussianNoiseMedical, RandomGamma


from DataClasses import *
from OrganLabels import *
from losses import DiceCELoss


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min', restore_best_weights=True, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

        self.monitor_op = np.less if mode == 'min' else np.greater
        self.best_loss = np.inf if mode == 'min' else -np.inf

    def __call__(self, current_loss, model):
        if self.monitor_op(current_loss - self.min_delta, self.best_loss):
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)

    def get_best_weights(self):
        return self.best_model_state


def main ():

    parser = argparse.ArgumentParser(description="Train Vanilla UNet on Slices with varying training set sizes")
    parser.add_argument('--Seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--Epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--BatchSize', type=int, default=32, help='Batch size for training')
    parser.add_argument('--TrainSize', type=int, default=150,help='Number of training samples to use')
    parser.add_argument('--DataSamp', type=str, default='Curated', help='Type of data sampling to use: Curated or RandomSelect')
    parser.add_argument('--UseEarlyStopping', type=bool, default=True, help='Whether to use early stopping during training')
    parser.add_argument('--LRFactor', type=float, default=0.1, help='Learning rate factor for reducing learning rate on plateau')
    parser.add_argument('--LRPatience', type=int, default=15, help='Patience for learning rate scheduler')
    parser.add_argument('--UseArgumentation', type=bool, default=False, help='Whether to use data augmentation')
    args = parser.parse_args()


    if args.UseEarlyStopping:
        print("Early stopping is enabled.")
    else:
        print("Early stopping is disabled.")


    

   
    if args.Seed is not None:
        torch.manual_seed(args.Seed)
        print(f"Set seed to {args.Seed}")
    else:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        print("No seed set. Using default seed 42.")

    if wandb.run is not None:  # inside a sweep
        args.BatchSize = wandb.config.BatchSize
        args.Epochs = wandb.config.Epochs
        args.TrainSize = wandb.config.TrainSize


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # directory of training data
    train_dataset = {}
    val_dataset = {}
    if args.DataSamp == "Curated":
        train_dir = f"../CandidateSliceData{args.TrainSize}New/train/"  

    
        planes = ["sagittal", "axial", "coronal"]
        
        
        for plane in planes:
            if args.UseArgumentation:
                # define augmentation transforms
                train_transform = T.Compose([
                    T.RandomApply([T.RandomRotation(degrees=15)], p=0.5),
                    T.RandomApply([T.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1))], p=0.4),
                    T.RandomApply([T.RandomResizedCrop(size=(256, 256), scale=(0.8, 1))], p=0.3),

                    GaussianNoiseMedical(sigma=1, p=0.5),
                    RandomGamma(gamma_range=(0.8, 1.2), p=0.4),
                    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),

                ])
                ds_train = CuratedSliceDatasetArgumented(train_dir, plane, transform=train_transform)
            else:
                ds_train = CuratedSliceDataset(train_dir, plane)
            
            train_dataset[plane] = ds_train

    
    # directory of training data
    data_dir = "../TotalSegmentator_v2_Full/TrainSubjects/"

    available_subjects = [d for d in glob.glob(os.path.join(data_dir, "s*")) if os.path.isdir(d)]
    subjects_names = [os.path.basename(d) for d in available_subjects]
    print(f"Total available subjects in data dir: {len(available_subjects)}")
        
    planes = ["sagittal", "axial", "coronal"]

    
    for plane in planes:
        ds = Slicedataset(data_dir, plane, subjects_names)
        num_subjects = len(ds)

        num_val = 500

        val_indices = random.sample(range(num_subjects), num_val)
        if args.DataSamp == "RandomSelect":
            num_train = args.TrainSize // len(planes)  # divide equally among planes

            train_indices = random.sample(
                list(set(range(num_subjects)) - set(val_indices)), num_train
            )
            if args.UseArgumentation:
                # define augmentation transforms
                train_transform = T.Compose([
                    T.RandomApply([T.RandomRotation(degrees=15)], p=0.5),
                    T.RandomApply([T.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1))], p=0.4),
                    T.RandomApply([T.RandomResizedCrop(size=(256, 256), scale=(0.8, 1))], p=0.3),

                    GaussianNoiseMedical(sigma=1, p=0.5),
                    RandomGamma(gamma_range=(0.8, 1.2), p=0.4),
                    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),

                ])
                ds_arg = SlicedatasetArgumentation(data_dir, plane, subjects_names, transform=train_transform)
                train_subset = torch.utils.data.Subset(ds_arg, train_indices)
            else:
                train_subset = torch.utils.data.Subset(ds, train_indices)

            train_dataset[plane] = train_subset

        val_subset = torch.utils.data.Subset(ds, val_indices)
        val_dataset[plane] = val_subset
        

    # print the classes found in each dataset
    for plane in planes:
        train_classes = set()
        for i in range(len(train_dataset[plane])):
            sample = train_dataset[plane][i]
            mask = sample['mask'].numpy()
            unique_classes = np.unique(mask)
            train_classes.update(unique_classes.tolist())
        print(f"Training dataset classes in {plane} plane: {sorted(train_classes)}")
        # print warning if any class is missing
        missing_classes = set(GROUP_MAP.values()) - train_classes
        if missing_classes:
            print(f"Warning: Missing classes in training dataset for {plane} plane: {sorted(missing_classes)}")

        val_classes = set()
        for i in range(len(val_dataset[plane])):
            sample = val_dataset[plane][i]
            mask = sample['mask'].numpy()
            unique_classes = np.unique(mask)
            val_classes.update(unique_classes.tolist())
        print(f"Validation dataset classes in {plane} plane: {sorted(val_classes)}")
        # print warning if any class is missing
        missing_classes = set(GROUP_MAP.values()) - val_classes
        if missing_classes:
            print(f"Warning: Missing classes in validation dataset for {plane} plane: {sorted(missing_classes)}")

    #print how many pixels of each class in training set
    total_pixels = 0
    class_counts = {k: 0 for k in GROUP_MAP.values()}
    for plane, dataset in train_dataset.items():
        for i in range(len(dataset)):
            sample = dataset[i]
            mask = sample["mask"]
            total_pixels += mask.numel()
            for class_idx in class_counts.keys():
                class_counts[class_idx] += torch.sum(mask == class_idx).item()
    print("Training set class distribution:")
    for class_idx, count in class_counts.items():
        percentage = (count / total_pixels) * 100
        print(f"Class {class_idx}: {count} pixels ({percentage:.4f}%)")
    
    
    
    # combine datasets from different planes
    train_dataset = ConcatDataset(list(train_dataset.values()))
    val_dataset = ConcatDataset(list(val_dataset.values()))
    print(f"Training subjects: {len(train_dataset)} slices")
    print(f"Validation subjects: {len(val_dataset)} slices")

    num_train = args.TrainSize
    
    
    # create data loader
    batch_size = args.BatchSize
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.LRPatience, factor=args.LRFactor, min_lr=1e-7)

    early_stopping = EarlyStopping(patience=args.LRPatience+5, verbose=True) if args.UseEarlyStopping else None
    
    print(model)

    mainpath = f"TRASHVanillaUNet_model_{args.DataSamp}_{args.LRFactor}_{args.LRPatience}_Argumentation/"
    subpath  = f"TrainSize{num_train}/BatchSize{batch_size}/"
    modelpath = os.path.join(mainpath, subpath)
    os.makedirs(modelpath, exist_ok=True)

    modelpath = os.path.join(modelpath, "best_model.pth")


    num_epochs = args.Epochs

    name = "VanillaUNet_2D_fullDataset_TrainSize_" + str(num_train) + "_BatchSize_" + str(batch_size)
    wandb.init(
        project="SmallDataTraining",
        name=name,
        group="DINOv3_skip",
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
            "TrainSize": num_train,
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
        #  TRAIN 
        model.train()
        ce_running = 0.0
        dice_running = 0.0
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            inputs = batch["image"].to(device).squeeze(1)  # [B, C, H, W]
            labels = batch["mask"].to(device).squeeze(1)  # [B, H, W]

            torch.cuda.reset_peak_memory_stats()

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast('cuda'):
                outputs = model(inputs)
                loss, ce_loss, dice_loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

          

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

        #  VALIDATION 
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

                    with amp.autocast('cuda'):
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
            
            prev_lr = optimizer.param_groups[0]['lr']

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # RESET early stopping if LR was reduced
            if current_lr < prev_lr:
                print("LR reduced — resetting early stopping counter")
                early_stopping.counter = 0

            
            wandb.log({
                "epoch": epoch + 1,
                "val/loss": avg_val_loss,
                "val/ce_loss": avg_val_ce,
                "val/dice_loss": avg_val_dice,
                "learning_rate": current_lr
            })

            
            print(f"Epoch {epoch+1}/{num_epochs}, "
                f"Validation Loss: {avg_val_loss:.4f}, CE: {avg_val_ce:.4f}, Dice: {avg_val_dice:.4f}")

            # save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), modelpath)
                print(f"Best model saved to path: {modelpath}")

                wandb.log({"best_val_loss": best_val_loss})
                wandb.save(modelpath)
            
            # early stopping check
            if early_stopping is not None:
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered. Ending training.")
                    break

    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print(f"Total training time: {elapsed/60:.2f} minutes")
    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()



    
