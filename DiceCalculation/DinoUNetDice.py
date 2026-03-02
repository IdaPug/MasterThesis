import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import pandas as pd
import os
import random
import nibabel as nib
from dataclasses import dataclass
from DataClasses import *
from Decoder import DINOv3UNetEncodeDecoder, DINOv3UNetEncodeDecoderV2, DINOv3UNetEncodeDecoderAttentionGate, DINOv3UNetEncodeDecoderAttentionGateFullV1, DINOv3UNetEncodeDecoderAttentionGateFullV2
from OrganLabels import *
from torchvision import transforms
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
import argparse
import psutil
import gc

# Helper for memory monitoring
def print_mem(tag=""):
    ram = psutil.virtual_memory().used / 1e9
    gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[{tag}] RAM: {ram:.2f} GB | GPU: {gpu:.2f} GB", flush=True)

# Metadata dataclass
@dataclass
class NiftiMeta:
    affine: np.ndarray
    header: nib.nifti1.Nifti1Header
    voxel_spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    shape: tuple[int, int, int]

# Utility functions
def load_nifti(path: str):
    img = nib.load(path)
    data = img.get_fdata()
    meta = NiftiMeta(
        affine=img.affine,
        header=img.header,
        voxel_spacing=img.header.get_zooms()[:3],
        orientation=nib.aff2axcodes(img.affine),
        shape=img.shape,
    )
    return data, meta

def plotprediction(image, mask, prediction, subject_id=None, dice_score=None, save_dir="predictions"):
    num_labels = max(GROUP_MAP.values()) + 1
    colors = plt.cm.jet(np.linspace(0, 1, num_labels))
    colors = np.vstack([[0, 0, 0, 0], colors])
    cmap = ListedColormap(colors)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap="gray")
    axs[0].imshow(mask, cmap=cmap, alpha=0.5)
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(image, cmap="gray")
    axs[1].imshow(prediction, cmap=cmap, alpha=0.5)
    axs[1].set_title("Prediction")
    axs[1].axis("off")

    for label_idx in range(1, num_labels):
        axs[2].bar(0, 0, color=colors[label_idx], label=label_names.get(label_idx, f"Label {label_idx}"))
    title = f"Subject: {subject_id}"
    if dice_score is not None:
        title += f" | DICE: {dice_score:.4f}"
    axs[2].set_title(title)
    axs[2].legend(loc="center", bbox_to_anchor=(1, 0.5))
    axs[2].axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"prediction_{subject_id}.png"))
    plt.close()

def dice_score(gt, pred, label):
    gt_mask = (gt == label).astype(np.int32)
    pred_mask = (pred == label).astype(np.int32)
    intersection = np.sum(gt_mask * pred_mask)
    denom = np.sum(gt_mask) + np.sum(pred_mask)
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / denom

def dice_per_label(gt, pred, num_classes, background=0):
    scores = []
    for label in range(num_classes):
        if label == background:
            continue
        scores.append(dice_score(gt, pred, label))
    return np.array(scores)

def reconstruct_volume(predictions_list, orientation):
    probs = torch.from_numpy(np.stack(predictions_list))
    probs = probs.permute(1, 0, 2, 3)
    if orientation == "axial":
        vol = probs
    elif orientation == "coronal":
        vol = probs.permute(0, 2, 1, 3)
    elif orientation == "sagittal":
        vol = probs.permute(0, 2, 3, 1)
    else:
        raise ValueError("Invalid orientation")
    return vol

def resample_to_reference(prob_map, ref_shape):
    prob_map = prob_map.unsqueeze(0)
    prob_resampled = F.interpolate(prob_map, size=ref_shape, mode="trilinear", align_corners=False)
    return prob_resampled.squeeze(0)

def fuse_probabilities(p_axial, p_coronal, p_sagittal, weights=(1/3, 1/3, 1/3)):
    w = torch.tensor(weights).view(3, 1, 1, 1, 1).to(p_axial.device)
    return w[0]*p_axial + w[1]*p_coronal + w[2]*p_sagittal

# Argparse & setup

def main():
    parser = argparse.ArgumentParser(description="DINOv3 UNet testing with per-subject fusion")
    parser.add_argument("--dinomodel", type=str, default="dinov3_vits16", help="DINO model name")
    parser.add_argument("--model", type=str, default="v1", help="Model name")
    parser.add_argument('--Model_path', type=str, default="", help="Model path")
    parser.add_argument('--LRType', type=str, default="constant", help='Learning rate schedule')
    parser.add_argument('--TrainSizePercent', type=float, default=10.0, help='Percentage of subjects to use for training of of total number (0-1)')
    parser.add_argument('--TestSize', type=int, default=200, help='Number of subjects to use for testing')
    
    args = parser.parse_args()

    modelPaths = {
        'dinov3_vits16': '../../DinoWeights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': '../../DinoWeights/dinov3_vits16plus.pth',
        'dinov3_vitb16': '../../DinoWeights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': '../../DinoWeights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vith16plus': '../../DinoWeights/dinov3_vith16plusw.pth'
    }

    # check if the weight path exist
    curr_path = modelPaths[args.dinomodel]
    if not os.path.exists(curr_path):
        raise FileNotFoundError(f"Weight file for model {args.dinomodel} not found at path: {curr_path}")
    else:
        print(f"Found weight file for model {args.dinomodel} at path: {curr_path}")


    dinov3_model = torch.hub.load("../dinov3", args.dinomodel, source="local", weights=modelPaths[args.dinomodel])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov3_model = dinov3_model.to(device)
    print(f"Loaded {args.model} on {device}")

     # build a directory of embed_dim based on which dino
    dim_dicts = {
        'dinov3_vits16': 384,
        'dinov3_vits16plus': 384,
        'dinov3_vitb16': 768,
        'dinov3_vitl16': 1024,
        'dinov3_vith16plus': 1080
    }

    target_size = (256, 256)


    # dir for test
    data_dir = "../../TotalSegmentator_v2_Full/TestSubjects/"

    available_subjects = [d for d in glob.glob(os.path.join(data_dir, "s*")) if os.path.isdir(d)]
    print(f"Total available test subjects in data dir: {len(available_subjects)}")

    # select test subjects randomly
    random.seed(42)
    random.shuffle(available_subjects)
    testsubjects = [os.path.basename(d) for d in available_subjects[:args.TestSize]]
    print(f"Selected {len(testsubjects)} test subjects.")
    #testsubjects = ['s0727']

    Numplot = int(args.TestSize * 0.1)
    plotevery = max(1, len(testsubjects) // max(1, Numplot))

    layers = [2, 5, 8, 11]
    planes = ["axial", "sagittal", "coronal"]


    samp = VolSlicedataset(data_dir, planes[0], [testsubjects[0]])[0]
    testimg = samp["images"][0, :, :, :].to(device)
    print(f"Sample input shape: {testimg.shape}")
    in_channels = testimg.shape[1]
    num_classes = max(GROUP_MAP.values()) + 1

    # model is v1 or v2:
    if args.model == "v1":
        embed_dim = dim_dicts[args.dinomodel]*2
        model = DINOv3UNetEncodeDecoder(
        in_channels=in_channels,
        embed_dim=embed_dim,
        dinov3_model=dinov3_model,
        n_layers=layers,
        num_classes=num_classes,
        out_size=target_size
    )
    elif args.model == "v2":
        embed_dim = dim_dicts[args.dinomodel]
        model = DINOv3UNetEncodeDecoderV2(
        in_channels=in_channels,
        dino_dim=embed_dim,
        dinov3_model=dinov3_model,
        n_layers=layers,
        num_classes=num_classes,
        out_size=target_size
    )
    elif args.model == "AG":
        dino_dim = dim_dicts[args.dinomodel]
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
        dino_dim = dim_dicts[args.dinomodel]*2
        model = DINOv3UNetEncodeDecoderAttentionGateFullV1(
            in_channels=in_channels,
            embed_dim=dino_dim,
            dinov3_model=dinov3_model,
            n_layers=4,
            num_classes=num_classes,
            out_size=(256, 256)     
        )
    elif args.model == "AG_fullV2":
        dino_dim = dim_dicts[args.dinomodel]
        model = DINOv3UNetEncodeDecoderAttentionGateFullV2(
            in_channels=in_channels,
            dino_dim=dino_dim,
            dinov3_model=dinov3_model,
            embed_dim=1024,
            n_layers=4,
            num_classes=num_classes,
            out_size=(256, 256)     
        )

    model_path = f"../{args.Model_path}/dinomodel_{args.dinomodel}/TrainSize{args.TrainSizePercent}_LR{args.LRType}/best_model.pth"

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    print(f"Using model: {model_path}")

    print(model)

    
    # output dirs
    savedir = os.path.join(f"../3DSegmentationAttGated", f"model/TrainSize{args.TrainSizePercent}_LR{args.LRType}")
    print(savedir)
    os.makedirs(savedir, exist_ok=True)
    savedir_plot = os.path.join("../predictionsAttGated",  f"model/TrainSize{args.TrainSizePercent}_LR{args.LRType}")
    os.makedirs(savedir_plot, exist_ok=True)

    # Main processing loop
    labelDices = {}
    count = 0
    for i, subject_id in enumerate(testsubjects):
        print(f"\n=== Processing Subject {subject_id} ({i+1}/{len(testsubjects)}) ===")
        subject_vol = {}
        print_mem(f"Start {subject_id}")

        count += 1
        if count == plotevery:
            do_plot = True
            count = 0
            print(f"  Will plot this subject. {subject_id}")
        else:
            do_plot = False

        for plane in planes:
            print(f"  Plane: {plane}")
            slice_ds = VolSlicedataset(data_dir, plane, [subject_id])
            selsub = slice_ds.get_subject(subject_id)
            all_predictions = []

            with torch.no_grad():
                # plot a sample slice prediction
                if do_plot:
                    sample_idx = selsub['images'].shape[0] // 2
                    slice_img = selsub['images'][sample_idx, :, :, :].to(device)
                    slice_mask = selsub['masks'][sample_idx, :, :]

                    
                    output = model(slice_img).squeeze(0)
                    pred_mask = torch.argmax(output, dim=0).cpu().numpy()
                    plotimg = slice_img.squeeze(0).cpu().numpy()
                    plotimg = plotimg[0,:,:]
                    plotdir = os.path.join(savedir_plot, plane)

                    dice_scores_vis = [dice_score(slice_mask.numpy(), pred_mask, g) for g in range(1, num_classes)]
                    mean_dice_vis = np.mean(dice_scores_vis)

                    plotprediction(
                        image=plotimg,
                        mask=slice_mask[0, :, :].numpy(),
                        prediction=pred_mask,
                        subject_id=subject_id,
                        dice_score=mean_dice_vis,
                        save_dir=plotdir
                    )

                    del slice_img, slice_mask, output, pred_mask
                    torch.cuda.empty_cache()   


                for slice_idx in range(selsub['images'].shape[0]):
                    slice_img = selsub['images'][slice_idx, :, :, :].to(device)
                    
                    output = model(slice_img).squeeze(0)
                    #print(output.shape)
                    
                    all_predictions.append(output.cpu().numpy())

                    del slice_img, output
                    torch.cuda.empty_cache()

            prob_vol = reconstruct_volume(all_predictions, plane)
            print(f"    Reconstructed prob volume shape: {prob_vol.shape}")
            subject_vol[plane] = prob_vol.cpu()
            del all_predictions, slice_ds, selsub
            torch.cuda.empty_cache()
            gc.collect()
            #print_mem(f"After plane {plane}")

        # Fuse and save immediately
        print(f"  Fusing subject {subject_id}")
        path = os.path.join(data_dir, subject_id)
        image, meta = load_nifti(os.path.join(path, "ct.nii.gz"))
        seg_path = os.path.join(path, "segmentations")
        mask = np.zeros_like(image, dtype=np.int16)
        for organ_path in glob.glob(os.path.join(seg_path, "*.nii.gz")):
            organ_name = os.path.splitext(os.path.splitext(os.path.basename(organ_path))[0])[0]
            label_id = OFFICIAL_LABELS[organ_name]
            organ_data = nib.load(organ_path).get_fdata()
            mask[organ_data > 0] = label_id

       

        out = np.zeros_like(mask, dtype=np.int16)
        for old_id, new_id in GROUP_MAP.items():
            out[mask == old_id] = new_id

        # unique labels in out
        #unique_labels = np.unique(out)
        #print(f"  Unique labels in ground truth: {unique_labels}")


        ref_shape = out.shape
        res_axial = resample_to_reference(subject_vol["axial"], ref_shape)
        res_coronal = resample_to_reference(subject_vol["coronal"], ref_shape)
        res_sagittal = resample_to_reference(subject_vol["sagittal"], ref_shape)


        print(f"  Resampled shapes: Axial {res_axial.shape}, Coronal {res_coronal.shape}, Sagittal {res_sagittal.shape}")
        
        fused_probs = fuse_probabilities(res_axial, res_coronal, res_sagittal)
        fused_seg = torch.argmax(fused_probs, dim=0).numpy().astype(np.uint8)
        print("unique labels in fused segmentation:", np.unique(fused_seg))

   

        unique_labels = np.unique(fused_seg)
        #print(f"  Unique labels in fused segmentation: {unique_labels}")
        do_plot = True
        dice_scores = {}
        for label in unique_labels:
            if label == 0:
                continue
            dice = dice_score(out, fused_seg, label)
            dice_scores[label] = dice
            organ_seg = (fused_seg == label).astype(np.uint8)
            if do_plot:
                subject_dir = os.path.join(savedir, subject_id)
                os.makedirs(subject_dir, exist_ok=True)
  
        labelDices[subject_id] = dice_scores
        mean_dice = np.mean(list(dice_scores.values()))
        print(f"  Subject Mean Dice: {mean_dice:.4f}")


        # cleanup
        del subject_vol, fused_probs, fused_seg, res_axial, res_coronal, res_sagittal, out, mask
        torch.cuda.empty_cache()


    # Final Dice summary

    labeldicedir = "../ThesisPlotting/LabelDices.txt"
    # make dataframe to save avarage dice score pr. class
    # check if file exist, if not create it with columns model, TrainSize, label1avarage dice, label2average dice, ...
    avg_dice_df = pd.read_csv(labeldicedir) if os.path.exists(labeldicedir) else pd.DataFrame(columns=["model", "TrainSize"] + [f"label_{i}_dice" for i in range(1, num_classes)])
    modelname = f"{args.model}_{args.dinomodel}"

    # for 1 row
    row_dict = {"model": modelname, "TrainSize": args.TrainSizePercent}



    print("\n=== Final Average Dice Scores per Label ===")
    final_dice_per_label = {}
    for subject_id, scores in labelDices.items():
        for label, score in scores.items():
            final_dice_per_label.setdefault(label, []).append(score)
    for label, scores in final_dice_per_label.items():
        avg_score = np.mean(scores)
        print(f"Average Dice for label {label} ({label_names[label]}): {avg_score:.4f}")
        # add to row dict
        row_dict[f"label_{label}_dice"] = avg_score


    avg_dice_df = pd.concat([avg_dice_df, pd.DataFrame([row_dict])], ignore_index=True)
    
    # save the average dice scores to txt file
    avg_dice_df.to_csv(labeldicedir, index=False)
    print(f"Saved average dice scores to {labeldicedir}")

    overall_mean_dice = np.mean([score for scores in labelDices.values() for score in scores.values()])
    print(f"\nOverall Mean Dice (excluding background): {overall_mean_dice:.4f}")

    # save the average dice scores to txt file
    avg_dice_df.to_csv(labeldicedir, index=False)
    print(f"Saved average dice scores to {labeldicedir}")

    
     
    # save to txt file
    plottingmodel = args.model + "ALT_" + args.LRType
    if args.dinomodel == 'dinov3_vits16':
        file_dir = "../ThesisPlotting/DINOUNETFINAL.txt"
    if args.dinomodel == 'dinov3_vitb16':
        file_dir = "../ThesisPlotting/DINOUNETFINAL_Large.txt"
    if args.dinomodel == 'dinov3_vitl16':
        file_dir = "../ThesisPlotting/DINOUNETFINAL_LargeLarger.txt"
    df = pd.read_csv(file_dir) if os.path.exists(file_dir) else pd.DataFrame(columns=["model", "TrainSize", "LRType", "dice"])



    print(plottingmodel, args.TrainSizePercent, overall_mean_dice)
    # print number of columns and rows
    print(f"DataFrame shape: {df.shape}")

    # Check if this model+percent already exists
    mask = (df['model'] == plottingmodel) & (df['TrainSize'] == args.TrainSizePercent)
    if mask.any():
        print("Entry exists, updating dice score.")
        df.loc[mask, 'dice'] = overall_mean_dice
    else:
        df = pd.concat([df, pd.DataFrame([[plottingmodel, args.TrainSizePercent, args.LRType, overall_mean_dice]], columns=df.columns)], ignore_index=True)

    df.to_csv(file_dir, index=False)
    print(f"Saved results to {file_dir}")


if __name__ == "__main__":
    main()