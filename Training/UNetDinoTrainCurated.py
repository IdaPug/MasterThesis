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
import torch.nn.functional as F
import timeit
from torch import amp
import gc
import random
import pandas as pd
from torchvision.transforms import v2 as T

scaler = amp.GradScaler('cuda')


from Decoder import DINOv3UNetEncodeDecoder, DINOv3UNetEncodeDecoderWithSkips, DINOv3UNetEncodeDecoderV2, DINOv3UNetEncodeDecoderAttentionGateFullV1, DINOv3UNetEncodeDecoderAttentionGateFullV2
from DataClasses import *
from OrganLabels import *
from losses import DiceCELoss, WeightedDiceCELoss
from ArgumentaitonClasses import GaussianNoiseMedical, RandomGamma

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

# class_freq: tensor of shape [C], includes background
# values are fraction of voxels per class
# e.g. [0.85, 0.10, 0.04, 0.01]

def compute_dice_weights(class_freq, eps=1e-6):
    weights = 1.0 / torch.log(1.02 + class_freq + eps)
    weights = weights / weights.sum()
    return weights


def main():
 
    target_size = (256, 256)

    parser = argparse.ArgumentParser(description="Test different DINO models for feature extraction for training UNet decoder.")
    parser.add_argument('--DinoModel', type=str, default='dinov3_vits16', help='DINO model to use for feature extraction')
    parser.add_argument('--model', type=str, default='v1', help='DINO model to use for feature extraction')
    parser.add_argument('--Seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--Epochs', type=int, default=200, help='Number of training epochs')
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
        print("No seed set. Defaulting to 42.")

    if wandb.run is not None:  # inside a sweep
        args.DinoModel = wandb.config.DinoModel
        args.model = wandb.config.model
        args.Epochs = wandb.config.epochs
        args.BatchSize = wandb.config.batch_size
        args.TrainSize = wandb.config.TrainSize



    modelPaths = {
        'dinov3_vits16': '../DinoWeights/dinov3_vits16.pth',
        'dinov3_vits16plus': '../DinoWeights/dinov3_vits16plus.pth',
        'dinov3_vitb16': '../DinoWeights/dinov3_vitb16.pth',
        'dinov3_vitl16': '../DinoWeights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vith16plus': '../DinoWeights/dinov3_vith16plusw.pth'
    }

    
    dinomodel = args.DinoModel
    # load DINOv3 model
    if dinomodel not in modelPaths:
        raise ValueError(f"Model {dinomodel} not recognized. Available models: {list(modelPaths.keys())}")

    dinov3_model = torch.hub.load('dinov3', dinomodel, source='local', weights=modelPaths[dinomodel])
    print(f"Loaded DINO model: {dinomodel}")
    
    # build a directory of embed_dim based on which dino
    dim_dicts = {
        'dinov3_vits16': 384,
        'dinov3_vits16plus': 384,
        'dinov3_vitb16': 768,
        'dinov3_vitl16': 1024,
        'dinov3_vith16plus': 1080
    }

    # freeze encoder
    for param in dinov3_model.parameters():
        param.requires_grad = False
    dinov3_model.eval() 

    # get dino to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dinov3_model = dinov3_model.to(device)
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
    
    # compute class frequencies
    class_freq = torch.tensor([class_counts[k] for k in sorted(class_counts.keys())], dtype=torch.float32) / total_pixels
    print("Class frequencies:", class_freq.numpy())
    # compute dice weights
    dice_weights = compute_dice_weights(class_freq)
    print("Computed Dice weights:", dice_weights.numpy())
    
    # combine datasets from different planes
    train_dataset = ConcatDataset(list(train_dataset.values()))
    val_dataset = ConcatDataset(list(val_dataset.values()))
    print(f"Training dataset size: {len(train_dataset)} slices")
    print(f"Validation dataset size: {len(val_dataset)} slices")
    
    
    num_train = args.TrainSize
    
    # create data loader
    batch_size = args.BatchSize
    train_loader =DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    layers = [2,5,8,11]
    #layers = [1,4,8,11]

    # build model
    samp = train_dataset[0]
    testimg = samp["image"].to(device)
    print(testimg.shape)

    
    
    maskimg = samp['mask'].to(device)  # (1, H, W)
  

    # 
    in_channels = testimg.shape[1]
    num_classes = max(GROUP_MAP.values()) + 1

    
    # model is v1 or v2:
    if args.model == "v1":
        embed_dim = dim_dicts[dinomodel]*2
        model = DINOv3UNetEncodeDecoder(
        in_channels=in_channels,
        embed_dim=embed_dim,
        dinov3_model=dinov3_model,
        n_layers=layers,
        num_classes=num_classes,
        out_size=target_size
    )
    elif args.model == "v2":
        embed_dim = dim_dicts[dinomodel]
        model = DINOv3UNetEncodeDecoderV2(
        in_channels=in_channels,
        dino_dim=embed_dim,
        dinov3_model=dinov3_model,
        n_layers=layers,
        num_classes=num_classes,
        out_size=target_size
    )
    elif args.model == "AG_fullV1":
        dino_dim = dim_dicts[dinomodel]*2
        model = DINOv3UNetEncodeDecoderAttentionGateFullV1(
            in_channels=in_channels,
            embed_dim=dino_dim,
            dinov3_model=dinov3_model,
            n_layers=4,
            num_classes=num_classes,
            out_size=(256, 256)     
        )
    elif args.model == "AG_fullV2":
        dino_dim = dim_dicts[dinomodel]
        model = DINOv3UNetEncodeDecoderAttentionGateFullV2(
            in_channels=in_channels,
            dino_dim=dino_dim,
            dinov3_model=dinov3_model,
            embed_dim=1024,
            n_layers=4,
            num_classes=num_classes,
            out_size=(256, 256)     
        )

    model = model.to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    
    
    criterion = DiceCELoss().to(device)
    lr = 1e-4
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.LRPatience, factor=args.LRFactor, min_lr=1e-7)

    early_stopping = EarlyStopping(patience=args.LRPatience+5, verbose=True) if args.UseEarlyStopping else None

    print(model)

   


    if args.model == "v1":
        mainpath = f"TRASHTrainings2D_UnetDino_v1_{args.DataSamp}_{args.LRFactor}_{args.LRPatience}/"
        subpath  = f"dinomodel_{dinomodel}/TrainSize{num_train}/BatchSize{args.BatchSize}/"
        modelpath = os.path.join(mainpath, subpath)
        os.makedirs(modelpath, exist_ok=True)
    elif args.model == "v2":
        mainpath = f"Trainings2D_UnetDino_v2_{args.DataSamp}_{args.LRFactor}_{args.LRPatience}/"
        subpath  = f"dinomodel_{dinomodel}/TrainSize{num_train}/BatchSize{args.BatchSize}/"
        modelpath = os.path.join(mainpath, subpath)
        os.makedirs(modelpath, exist_ok=True)
    elif args.model == "AG_fullV1":
        mainpath = f"Trainings2D_UnetDino_AG_fullV1_{args.DataSamp}_{args.LRFactor}_{args.LRPatience}/"
        subpath  = f"dinomodel_{dinomodel}/TrainSize{num_train}/BatchSize{args.BatchSize}/"
        modelpath = os.path.join(mainpath, subpath)
        os.makedirs(modelpath, exist_ok=True)
    elif args.model == "AG_fullV2":
        mainpath = f"TRASHTrainings2D_UnetDino_AG_fullV2_{args.DataSamp}_{args.LRFactor}_{args.LRPatience}_Argumentation/"
        subpath  = f"dinomodel_{dinomodel}/TrainSize{num_train}/BatchSize{args.BatchSize}/"
        modelpath = os.path.join(mainpath, subpath)
        os.makedirs(modelpath, exist_ok=True)

    modelpath = os.path.join(modelpath, "best_model.pth")

    num_epochs = args.Epochs
    

    name = f"DINOV3_{dinomodel}_{args.model}_Data{args.DataSamp}_TrainSize{num_train}_BS{batch_size}__{args.LRFactor}_{args.LRPatience}"
    wandb.init(
        project="SmallDataTraining",
        name=name,
        group=f"DINOV3_skip_{dinomodel}",
        config={
        "model": args.model,
        "TrainSize":  num_train,
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "loss": "Dice + CrossEntropy",
    }
    )   

    start_time = timeit.default_timer()
    print("Starting training...")
    val_every = 1  # validate every n epochs
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
            #inputs = batch["image"].to(device)
            inputs = batch["image"].to(device).squeeze(1)  # [B, H, W]
            labels = batch["mask"].to(device).squeeze(1)  # [B, H, W]

            #print(inputs.shape)
            #print(labels.shape)
            
            torch.cuda.reset_peak_memory_stats()
            
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast('cuda'):
                outputs = model(inputs)
                loss, ce_loss, dice_loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #print("Computed loss")

            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory: {peak_mem:.2f} GB")
            
            total_loss += loss.item()
            ce_running += ce_loss
            dice_running += dice_loss
            total_batches += 1

            del inputs, labels, outputs, loss, ce_loss, dice_loss
            gc.collect()
            torch.cuda.empty_cache()

            print(f"processed batch {total_batches} in epoch {epoch+1}")

            if total_batches % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"[Epoch {epoch+1} | Batch {total_batches}] "
                    f"GPU allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


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
                    inputs = batch["image"].to(device).squeeze(1)  # [B, H, W]
                    labels = batch["mask"].to(device).squeeze(1)  # [B, H, W]

                    with amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss, ce_loss, dice_loss = criterion(outputs, labels)


                    loss_val_total += loss.item()
                    ce_val_total += ce_loss
                    dice_val_total += dice_loss
                    val_batches += 1

                    del inputs, labels, outputs, loss, ce_loss, dice_loss
                    gc.collect()
                    torch.cuda.empty_cache()
        

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
                
        torch.cuda.empty_cache()

    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print(f"Total training time: {elapsed/60:.2f} minutes")
    wandb.log({"training_time_minutes": elapsed/60})
    print("Training complete.")

    # # write trainin time to file
    # # save to txt file
    # file_dir = "ThesisPlotting/DINOUNetTime.txt"
    # df = pd.read_csv(file_dir) if os.path.exists(file_dir) else pd.DataFrame(columns=["model", "percent", "time"])

    # # Check if this model+percent already exists 
    # mask = (df['model'] == args.model) & (df['time'] == args.TrainSizePercent)
    # if mask.any():
    #     df.loc[mask, 'time'] = elapsed/60
    # else:
    #     df = pd.concat([df, pd.DataFrame([[args.model, args.TrainSizePercent, elapsed/60]], columns=df.columns)], ignore_index=True)

    # df.to_csv(file_dir, index=False)
    # print(f"Saved results to {file_dir}")



    wandb.finish()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main()






