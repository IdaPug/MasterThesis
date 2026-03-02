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

scaler = amp.GradScaler('cuda')


from Decoder import DINOv3UNetEncodeDecoder, DINOv3UNetEncodeDecoderV2, DINOv3UNetEncodeDecoderAttentionGate, DINOv3UNetEncodeDecoderAttentionGateFullV1, DINOv3UNetEncodeDecoderAttentionGateFullV2
from DataClasses import *
from OrganLabels import *
from losses import DiceCELoss


def main():
 
    target_size = (256, 256)

    parser = argparse.ArgumentParser(description="Test different DINO models for feature extraction for training UNet decoder.")
    parser.add_argument('--dino_model', type=str, default='dinov3_vits16', help='DINO model to use for feature extraction')
    parser.add_argument('--model', type=str, default='v1', help='DINO model to use for feature extraction')
    parser.add_argument('--TrainSizePercent', type=float, default=0.1, help='Percentage of subjects to use for training of of total number (0-1)')
    parser.add_argument('--Seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--Epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--LRMethod', type=str, default='constant', help='Learning rate method: constant or onecycle')
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
        print("No seed set. Defaulting to 42.")

    if wandb.run is not None:  # inside a sweep
        args.TrainSizePercent = wandb.config.TrainSizePercent
        args.model = wandb.config.model
        args.Epochs = wandb.config.epochs
        args.LRMethod = wandb.config.LRMethod



    # make sure that LRMethod is valid
    if args.LRMethod not in ["constant", "ReduceLROnPlateau", "CosineAnnealing"]:
        raise ValueError("LRMethod must be one of: constant, ReduceLROnPlateau, CosineAnnealing")
    


    modelPaths = {
        'dinov3_vits16': '../DinoWeights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': '../DinoWeights/dinov3_vits16plus.pth',
        'dinov3_vitb16': '../DinoWeights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': '../DinoWeights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vith16plus': '../DinoWeights/dinov3_vith16plusw.pth'
    }

    dinomodel = args.dino_model
    # check if the weight path exist
    curr_path = modelPaths[dinomodel]
    if not os.path.exists(curr_path):
        raise FileNotFoundError(f"Weight file for model {dinomodel} not found at path: {curr_path}")
    else:
        print(f"Found weight file for model {dinomodel} at path: {curr_path}")




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

        random_indices = random.sample(range(num_subjects), num_train)
        val_indices = random.sample(range(num_subjects), num_val)

        train_subset = torch.utils.data.Subset(ds, random_indices)
        val_subset = torch.utils.data.Subset(ds, val_indices)

        train_dataset[plane] = train_subset
        val_dataset[plane] = val_subset



    # combine datasets from different planes
    train_dataset = ConcatDataset(list(train_dataset.values()))
    val_dataset = ConcatDataset(list(val_dataset.values()))
    print(f"Training subjects: {len(train_dataset)} slices. {args.TrainSizePercent}% of total available.")
    print(f"Validation subjects: {len(val_dataset)} slices. 5% of total available.")

    
    # create data loader
    batch_size = 32
    train_loader =DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    layers = [2,5,8,11]
    #layers = [1,4,8,11]

    # build model
    samp = train_dataset[0]
    testimg = samp["image"].to(device)
    print(testimg.shape)

    
    maskimg = samp['mask'].to(device)  # (1, H, W)
    print(maskimg.dtype)
    print(maskimg.unique())
    print(maskimg.shape)
    
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
    elif args.model == "AG":
        dino_dim = dim_dicts[dinomodel]
        model = DINOv3UNetEncodeDecoderAttentionGate(
            in_channels=in_channels,
            dino_dim=dino_dim,
            dinov3_model=dinov3_model,
            embed_dim=1024,
            n_layers=4,
            num_classes=num_classes,
            out_size=(256, 256)
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    
    
    criterion = DiceCELoss().to(device)
    lr = 1e-4
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    if args.LRMethod == "constant":
        print("Using constant learning rate.")
        LRSTEP = False
    elif args.LRMethod == "ReduceLROnPlateau":
        print("Using ReduceLROnPlateau learning rate.")
        LRSTEP = True
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif args.LRMethod == "CosineAnnealing":
        print("Using CosineAnnealing learning rate.")
        LRSTEP = True
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epochs, eta_min=1e-6)
    print(model)


    if args.model == "v1":
        mainpath = "Trainings2D_UnetDino_deep_old/"
        subpath  = f"dinomodel_{dinomodel}/TrainSize{args.TrainSizePercent}_LR{args.LRMethod}/"
        modelpath = os.path.join(mainpath, subpath)
        os.makedirs(modelpath, exist_ok=True)
    elif args.model == "v2":
        mainpath = "Trainings2D_UnetDinoV2_deeper_old_alt/"
        subpath  = f"dinomodel_{dinomodel}/TrainSize{args.TrainSizePercent}_LR{args.LRMethod}/"
        modelpath = os.path.join(mainpath, subpath)
        os.makedirs(modelpath, exist_ok=True)
    elif args.model == "AG":
        mainpath = "Trainings2D_UnetDinoAG_deep_old/"
        subpath  = f"dinomodel_{dinomodel}/TrainSize{args.TrainSizePercent}_LR{args.LRMethod}/"
        modelpath = os.path.join(mainpath, subpath)
        os.makedirs(modelpath, exist_ok=True)
    elif args.model == "AG_fullV1":
        mainpath = "Trainings2D_UnetDinoAGFull_deep_old/"
        subpath  = f"dinomodel_{dinomodel}/TrainSize{args.TrainSizePercent}_LR{args.LRMethod}/"
        modelpath = os.path.join(mainpath, subpath)
        os.makedirs(modelpath, exist_ok=True)
    elif args.model == "AG_fullV2":
        mainpath = "Trainings2D_UnetDinoAGFullV2_deep_old/"
        subpath  = f"dinomodel_{dinomodel}/TrainSize{args.TrainSizePercent}_LR{args.LRMethod}/"
        modelpath = os.path.join(mainpath, subpath)
        os.makedirs(modelpath, exist_ok=True)
    
            
    
    modelpath = os.path.join(modelpath, "best_model.pth")

    num_epochs = args.Epochs


    name = f"DINOV3_UNetDino_{args.model}_TrainSize{args.TrainSizePercent}_LR{args.LRMethod}_Epochs{num_epochs}"
    wandb.init(
        project="UNetDinoTrain_const",
        name=name,
        group=f"DINOV3_skip_{dinomodel}",
        config={
        "model": args.model,
        "TrainSizePercent": args.TrainSizePercent,
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "loss": "Dice + CrossEntropy",
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

        # VALIDATION 
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
            # check if we need to step the scheduler
            if LRSTEP:
                if args.LRMethod == "ReduceLROnPlateau":
                    scheduler.step(avg_val_loss)
                elif args.LRMethod == "CosineAnnealing":
                    scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{num_epochs}, "
                f"Validation Loss: {avg_val_loss:.4f}, CE: {avg_val_ce:.4f}, Dice: {avg_val_dice:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "val/loss": avg_val_loss,
                "val/ce_loss": avg_val_ce,
                "val/dice_loss": avg_val_dice,
                'lr': current_lr
            })

            # save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), modelpath)
                print(f"Best model saved to path: {modelpath}")

                wandb.log({"best_val_loss": best_val_loss})
                wandb.save(modelpath)
                
        torch.cuda.empty_cache()

    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print(f"Total training time: {elapsed/60:.2f} minutes")
    wandb.log({"training_time_minutes": elapsed/60})
    print("Training complete.")

    # write trainin time to file
    # save to txt file
    file_dir = "ThesisPlotting/DINOUNetTime.txt"
    df = pd.read_csv(file_dir) if os.path.exists(file_dir) else pd.DataFrame(columns=["model", "percent", "time"])

    # Check if this model+percent already exists 
    mask = (df['model'] == args.model) & (df['time'] == args.TrainSizePercent)
    if mask.any():
        df.loc[mask, 'time'] = elapsed/60
    else:
        df = pd.concat([df, pd.DataFrame([[args.model, args.TrainSizePercent, elapsed/60]], columns=df.columns)], ignore_index=True)

    df.to_csv(file_dir, index=False)
    print(f"Saved results to {file_dir}")



    wandb.finish()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main()






