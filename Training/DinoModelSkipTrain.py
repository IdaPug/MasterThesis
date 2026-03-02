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



from Decoder import UnetDecoder, DINOv3UNetDecoder,DINOv3UNetDecoderAlternative, DINOv3UNetDecoderAlternative2, DINOv3UNetDecoderAlternative3
from DataClasses import *
from OrganLabels import *
from losses import DiceCELoss


def main():
 
    target_size = (256, 256)

    parser = argparse.ArgumentParser(description="Test different DINO models for feature extraction for training UNet decoder.")
    parser.add_argument('--model', type=str, default='dinov3_vits16', help='DINO model to use for feature extraction')
    parser.add_argument('--TrainSizePercent', type=float, default=10, help='Percentage of subjects to use for training of of total number (0-1)')
    parser.add_argument('--Seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--Epochs', type=int, default=200, help='Number of training epochs')
    args = parser.parse_args()

    # check that train size percent is valid
    if args.TrainSizePercent <= 0 or args.TrainSizePercent > 100:
        raise ValueError("TrainSizePercent must be between 0 and 100")

    if args.Seed is not None:
        torch.manual_seed(args.Seed)
        print(f"Set seed to {args.Seed}")
    else:
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        print("No seed set. Defaulting to 42.")

    if wandb.run is not None:  # inside a sweep
        args.TrainSizePercent = wandb.config.TrainSizePercent
        args.model = wandb.config.model



    modelPaths = {
        'dinov3_vits16': '../DinoWeights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': '../DinoWeights/dinov3_vits16plus.pth',
        'dinov3_vitb16': '../DinoWeights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': '../DinoWeights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vith16plus': '../DinoWeights/dinov3_vith16plusw.pth'
    }

    # check if the weight path exist
    curr_path = modelPaths[args.model]
    if not os.path.exists(curr_path):
        raise FileNotFoundError(f"Weight file for model {args.model} not found at path: {curr_path}")
    else:
        print(f"Found weight file for model {args.model} at path: {curr_path}")
    

    

    # load DINOv3 model
    if args.model not in modelPaths:
        raise ValueError(f"Model {args.model} not recognized. Available models: {list(modelPaths.keys())}")

    dinov3_model = torch.hub.load('dinov3', args.model, source='local', weights=modelPaths[args.model])
    print(f"Loaded DINO model: {args.model}")
    
    # freeze encoder
    for param in dinov3_model.parameters():
        param.requires_grad = False
    dinov3_model.eval() 

    # print the weights of the first linear layer of the first block to verify that the model is loaded correctly
    first_block_weights = dinov3_model.blocks[0].attn.qkv.weight.data
    print(f"First weights: {first_block_weights.flatten()[:5]}")  # print first 5 weights

    


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
        print(f"Total subjects in plane {plane}: {num_subjects}")
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

    # layers based on which model 
    if args.model in ['dinov3_vits16', 'dinov3_vits16plus','dinov3_vitb16']:
        layers = [2,5,8,11]
    if args.model in ['dinov3_vitl16','dinov3_vith16plus']:
        layers = [5,11,17,23]

    

    # build model
    samp = train_dataset[0]
    testimg = samp["image"].to(device)
    feat = dinov3_model.get_intermediate_layers(testimg, n=1,reshape=True,norm=0,return_class_token=False)

    embed_dim = feat[0].shape[1]
    num_skips = len(layers)-1
    base_ch = embed_dim
    num_classes = max(GROUP_MAP.values()) + 1

    model = DINOv3UNetDecoder(
        embed_dim=embed_dim,
        num_skips=num_skips,
        base_ch=base_ch,
        num_classes=num_classes,
        out_size=target_size
    )

    model = model.to(device)
    # try to compute a forward pass
    with torch.no_grad():
        skip_features = dinov3_model.get_intermediate_layers(
            testimg, n=layers, reshape=True, norm=0, return_class_token=False
        )
        skip_features = [sf.detach().to(device) for sf in skip_features]
        feat = skip_features[-1]
        skip_features = skip_features[:-1]
        outputs = model(feat, skip_features)
        print(f"Test forward pass successful. Output shape: {outputs.shape}")
    
    criterion = DiceCELoss().to(device)
    lr = 1e-4
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    
    

    mainpath = "Trainings2D_DINOv3/"
    subpath  = f"dinomodel_{args.model}/TrainSize{args.TrainSizePercent}/"
    modelpath = os.path.join(mainpath, subpath)
    os.makedirs(modelpath, exist_ok=True)

    modelpath = os.path.join(modelpath, "best_model.pth")

    num_epochs = args.Epochs


    name = f"DINOV3_skip_{args.model}_TrainSize{args.TrainSizePercent}"
    wandb.init(
        project="ModelTesting_DINOv3_skip",
        name=name,
        group=f"DINOV3_skip_{args.model}",
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
        # ---------------- TRAIN ----------------
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
            
            with torch.no_grad():
                skip_features = dinov3_model.get_intermediate_layers(
                    inputs, n=layers, reshape=True, norm=0, return_class_token=False
                )
                # Detach to be safe and move to device
                skip_features = [sf.detach().to(device) for sf in skip_features]
                feat = skip_features[-1]
                skip_features = skip_features[:-1]
            
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast('cuda'):
                outputs = model(feat, skip_features)
                loss, ce_loss, dice_loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #print("Computed loss")


            
            total_loss += loss.item()
            ce_running += ce_loss
            dice_running += dice_loss
            total_batches += 1

            del inputs, labels, outputs, skip_features, feat, loss, ce_loss, dice_loss
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

                    skip_features = dinov3_model.get_intermediate_layers(
                    inputs, n=layers, reshape=True, norm=0, return_class_token=False
                    )
                    skip_features = [sf.detach().to(device) for sf in skip_features]
                    feat = skip_features[-1]
                    skip_features = skip_features[:-1]

                    with amp.autocast('cuda'):
                        outputs = model(feat, skip_features)
                        loss, ce_loss, dice_loss = criterion(outputs, labels)


                    loss_val_total += loss.item()
                    ce_val_total += ce_loss
                    dice_val_total += dice_loss
                    val_batches += 1

                    del inputs, labels, outputs, skip_features, feat, loss, ce_loss, dice_loss
                    gc.collect()
                    torch.cuda.empty_cache()
        

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
                
        torch.cuda.empty_cache()

    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print(f"Total training time: {elapsed/60:.2f} minutes")
    wandb.log({"training_time_minutes": elapsed/60})
    print("Training complete.")

    # write trainin time to file
    # save to txt file
    file_dir = "ThesisPlotting/DinoModelsTime.txt"
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



