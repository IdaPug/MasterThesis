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
from Decoder import UnetDecoder, AdapterUnetDecoder, DINOv3UNetDecoder, DINOv3UNetDecoderAlternative, DINOv3UNetDecoderAlternative2, DINOv3UNetDecoderAlternative3
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
    parser.add_argument("--model", type=str, default="dinov3_vits16", help="Model name")
    parser.add_argument('--TrainSizePercent', type=float, default=0.1, help='Percentage of subjects to use for training of of total number (0-1)')
    parser.add_argument('--TestSize', type=int, default=200, help='Number of subjects to use for testing')
    parser.add_argument('--BatchSize', type=int, default=32, help='Batch size for training')
    parser.add_argument('--TrainSize', type=int, default=150,help='Number of training samples to use')
    parser.add_argument('--DataSamp', type=str, default='RandomSelect', help='Type of data sampling to use: Curated or RandomSelect')
    parser.add_argument('--LRFactor', type=float, default=0.5, help='Learning rate factor for reducing learning rate on plateau')
    parser.add_argument('--LRPatience', type=int, default=20, help='Patience for learning rate scheduler')
    args = parser.parse_args()

    modelPaths = {
        'dinov3_vits16': '../../DinoWeights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': '../../DinoWeights/dinov3_vits16plus.pth',
        'dinov3_vitb16': '../../DinoWeights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': '../../DinoWeights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vith16plus': '../../DinoWeights/dinov3_vith16plusw.pth'
    }

    # check if the weight path exist
    curr_path = modelPaths[args.model]
    if not os.path.exists(curr_path):
        raise FileNotFoundError(f"Weight file for model {args.model} not found at path: {curr_path}")
    else:
        print(f"Found weight file for model {args.model} at path: {curr_path}")
    
    

    if args.model not in modelPaths:
        raise ValueError(f"Model {args.model} not recognized")

    dinov3_model = torch.hub.load("../dinov3", args.model, source="local", weights=modelPaths[args.model])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov3_model = dinov3_model.to(device)
    print(f"Loaded {args.model} on {device}")

    # print the weights of the first linear layer of the first block to verify that the model is loaded correctly
    first_block_weights = dinov3_model.blocks[0].attn.qkv.weight.data
    print(f"First weights: {first_block_weights.flatten()[:5]}")  # print first 5 weights
    
    

    target_size = (256, 256)
    slice_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(256)
    ])

    mask_transform = transforms.Lambda(lambda mask: F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode="nearest"
    ).squeeze(0).squeeze(0).long())

    # print current directory
    print(f"Current directory: {os.getcwd()}")

    # dir for test
    data_dir = "../../TotalSegmentator_v2_Full/TestSubjects/"

    available_subjects = [d for d in glob.glob(os.path.join(data_dir, "s*")) if os.path.isdir(d)]
    print(f"Total available test subjects in data dir: {len(available_subjects)}")

    # select test subjects randomly
    random.seed(42)
    random.shuffle(available_subjects)
    testsubjects = [os.path.basename(d) for d in available_subjects[:args.TestSize]]
    print(f"Selected {len(testsubjects)} test subjects.")

    Numplot = int(args.TestSize * 0.1)
    plotevery = max(1, len(testsubjects) // max(1, Numplot))

    layers = [2, 5, 8, 11]
    planes = ["axial", "sagittal", "coronal"]


    samp = VolSlicedataset(data_dir, planes[0], [testsubjects[0]])[0]
    testimg = samp["images"][0, :, :, :].to(device)
    feat = dinov3_model.get_intermediate_layers(testimg, n=1, reshape=True, norm=0, return_class_token=False)
    embed_dim = feat[0].shape[1]
    num_skips = len(layers) - 1
    base_ch = embed_dim
    num_classes = max(GROUP_MAP.values()) + 1

    # layers based on which model 
    if args.model in ['dinov3_vits16', 'dinov3_vits16plus','dinov3_vitb16']:
        layers = [2,5,8,11]
    if args.model in ['dinov3_vitl16','dinov3_vith16plus']:
        layers = [5,11,17,23]



    model = DINOv3UNetDecoder(embed_dim=embed_dim, num_skips=num_skips, base_ch=base_ch,
                            num_classes=num_classes, out_size=target_size)
    print(model)
    
    modelpath = f"../Trainings2D_DINOv3/dinomodel_{args.model}/TrainSize{args.TrainSizePercent}"
    #model_path = f"../../Trainings2D_DINOv3_SizeTest_dinomodel_dinov3_vits16_TrainSize5.0_best_model.pth"
    num_train = args.TrainSize
    #num_train = args.TrainSizePercent
    mainpath = f"../Trainings2D_DINOv3_{args.DataSamp}_{args.LRFactor}_{args.LRPatience}/"
    subpath  = f"dinomodel_{args.model}/TrainSize{num_train}/BatchSize{args.BatchSize}"
    modelpath = os.path.join(mainpath, subpath)

    model_path = os.path.join(modelpath, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    print(f"Using model: {model_path}")

    # print last time the model was modified
    mod_time = os.path.getmtime(model_path)
    print(f"Model last modified: {pd.to_datetime(mod_time, unit='s')}")
    
    
    
    # output dirs
    savedir = os.path.join("../3DSegmSkip", f"dinomodel_{args.model}/TrainSize{num_train}")
    os.makedirs(savedir, exist_ok=True)
    savedir_plot = os.path.join("../3DSegmSkip", f"dinomodel_{args.model}/TrainSize{num_train}")
    os.makedirs(savedir_plot, exist_ok=True)

    # -----------------------------
    # Main processing loop
    # -----------------------------
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

        do_plot = True

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

                    skip_features = dinov3_model.get_intermediate_layers(
                        slice_img, n=layers, reshape=True, norm=0, return_class_token=False
                    )
                    skip_features = [feat.to(device) for feat in skip_features]
                    feat = skip_features[-1]
                    skip_features = skip_features[:-1]

                    output = model(feat, skip_features).squeeze(0)
                    pred_mask = torch.argmax(output, dim=0).cpu().numpy()
                    plotimg = slice_img.squeeze(0).cpu().numpy()
                    plotimg = plotimg[0,:,:]
                    plotdir = os.path.join(savedir_plot, plane)

                    dice_scores_vis = [dice_score(slice_mask.numpy(), pred_mask, g) for g in range(1, num_classes)]
                    mean_dice_vis = np.mean(dice_scores_vis)
                    print(f"    Sample slice DICE: {mean_dice_vis:.4f}")

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
                    skip_features = dinov3_model.get_intermediate_layers(
                        slice_img, n=layers, reshape=True, norm=0, return_class_token=False
                    )
                    skip_features = [feat.to(device) for feat in skip_features]
                    feat = skip_features[-1]
                    skip_features = skip_features[:-1]

                    output = model(feat, skip_features).squeeze(0)
                    all_predictions.append(output.cpu().numpy())

                    del slice_img, skip_features, feat, output
                    torch.cuda.empty_cache()

            prob_vol = reconstruct_volume(all_predictions, plane)
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

        ref_shape = out.shape
        res_axial = resample_to_reference(subject_vol["axial"], ref_shape)
        res_coronal = resample_to_reference(subject_vol["coronal"], ref_shape)
        res_sagittal = resample_to_reference(subject_vol["sagittal"], ref_shape)

        fused_probs = fuse_probabilities(res_axial, res_coronal, res_sagittal)
        fused_seg = torch.argmax(fused_probs, dim=0).numpy().astype(np.uint8)

        
        unique_labels = np.unique(fused_seg)
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
                nib.save(
                    nib.Nifti1Image(organ_seg, affine=meta.affine, header=meta.header),
                    os.path.join(subject_dir, f"fused_seg_label_{label}_{label_names[label]}.nii.gz")
                )

        labelDices[subject_id] = dice_scores
        mean_dice = np.mean(list(dice_scores.values()))
        print(f"  Subject Mean Dice: {mean_dice:.4f}")
        #print_mem(f"End {subject_id}")

        # cleanup
        del subject_vol, fused_probs, fused_seg, res_axial, res_coronal, res_sagittal, out, mask
        torch.cuda.empty_cache()


    # Final Dice summary
    labeldicedir = "../ThesisPlotting/LabelDices.txt"
    # check if file exist, if not create it with columns model, TrainSize, label1avarage dice, label2average dice, ...
    avg_dice_df = pd.read_csv(labeldicedir) if os.path.exists(labeldicedir) else pd.DataFrame(columns=["model", "TrainSize"] + [f"label_{i}_dice" for i in range(1, num_classes)])
    modelname = "DinoEnc" + args.model

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

    
    
    
    LRType = 'constant'
    # save to txt file
    plottingmodel = "DinoEnvALT2" + "_" + LRType
    if args.model == 'dinov3_vits16':
        file_dir = "../ThesisPlotting/DINOUNETFINAL.txt"
    if args.model == 'dinov3_vitb16':
        file_dir = "../ThesisPlotting/DINOUNETFINAL_Large.txt"
    if args.model == 'dinov3_vitl16':
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
        df = pd.concat([df, pd.DataFrame([[plottingmodel, args.TrainSizePercent, LRType, overall_mean_dice]], columns=df.columns)], ignore_index=True)

    df.to_csv(file_dir, index=False)
    print(f"Saved results to {file_dir}")

    df.to_csv(file_dir, index=False)
    print(f"Saved results to {file_dir}")



if __name__ == "__main__":
    main()