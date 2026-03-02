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
from Decoder import *
from OrganLabels import *
from torchvision import transforms
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
import argparse
import psutil
import gc
import matplotlib.patches as mpatches


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


def blend_with_gray(color, gray=0.5, alpha=0.7):
    base = np.array([gray, gray, gray])
    return alpha * np.array(color[:3]) + (1 - alpha) * base

# Argparse & setup

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="DINOv3 UNet testing with per-subject fusion")
    
    parser.add_argument("--dinomodel", type=str, default="dinov3_vits16", help="DINOv3 model name")
    parser.add_argument('--TrainSizePercent', type=float, default=2.0, help='Percentage of subjects to use for training of of total number (0-1)')
    parser.add_argument('--Subject' ,type=str, default=None, help='Specific subject ID to test (overrides TestSize)')
    args = parser.parse_args()


    DinoEncPath = f"Trainings2D_DINOv3/dinomodel_{args.dinomodel}/TrainSize{args.TrainSizePercent}/best_model.pth"
    VanillaUnetPath = f"VanillaUNet_model_BetterData/TrainSize{args.TrainSizePercent}/best_model.pth"
    v1path = f"Trainings2D_UnetDino_deep_old/dinomodel_{args.dinomodel}/TrainSize{args.TrainSizePercent}_LRconstant/best_model.pth"
    AGv2Path = f"Trainings2D_UnetDinoAGFullV2_deep_old/dinomodel_{args.dinomodel}/TrainSize{args.TrainSizePercent}_LRconstant/best_model.pth"

    if args.TrainSizePercent == 150.0:
        DinoEncPath = f"Trainings2D_DINOv3_RandomSelect_0.5_20/dinomodel_{args.dinomodel}/TrainSize150/BatchSize32/best_model.pth"
        VanillaUnetPath = f"VanillaUNet_model_RandomSelect_0.5_20/TrainSize150/BatchSize32/best_model.pth"
        v1path = f"Trainings2D_UnetDino_v1_RandomSelect_0.2_20/dinomodel_{args.dinomodel}/TrainSize150/BatchSize32/best_model.pth"
        AGv2Path = f"Trainings2D_UnetDino_AG_fullV2_RandomSelect_0.2_20_Argumentation/dinomodel_{args.dinomodel}/TrainSize150/BatchSize32/best_model.pth"


    save3dpath = "Predictions3D/"
    
    planes = ["axial", "sagittal", "coronal"]

    data_dir = "../TotalSegmentator_v2_Full/TestSubjects/"
    available_subjects = [d for d in glob.glob(os.path.join(data_dir, "s*")) if os.path.isdir(d)]

    

    # check is subjet is given
    if args.Subject is not None:
        if args.Subject not in [os.path.basename(d) for d in available_subjects]:
            raise ValueError(f"Subject {args.Subject} not found in data directory.")
        else:
            print(f"Testing on specific subject: {args.Subject}")
    else:
        # set subjects as the first subject in available subjects for testing
        args.Subject = os.path.basename(available_subjects[0])
        
        print(f"No specific subject given. Defaulting to first available subject: {args.Subject}")

    # make 3x6 plot
    fig, axs = plt.subplots(3, 6, figsize=(20, 10))



    fig.suptitle(
        #f"Segmentation Results – Subject {args.Subject} – TrainSize % {args.TrainSizePercent}",
        f"Segmentation Results – Subject {args.Subject} – TrainSet of 150 Slices",
        fontsize=20,
        fontweight='bold',
        y=0.98  # controls vertical position
    )

    # stuff for plotting
    num_labels = max(GROUP_MAP.values()) + 1
    colors = plt.cm.jet(np.linspace(0, 1, num_labels-1))

    colors = np.vstack([[0, 0, 0, 0], colors])
    cmap = ListedColormap(colors)


    # print the rgb range 0-255  values of the label color mapping
    print("Label ID to RGB Color Mapping:")
    for label_id in sorted(label_names.keys()):
        color = cmap(label_id)
        color3 = color[:3] 

        print(f"Label ID {label_id}: {label_names[label_id]} \n")
        for i, c in enumerate(color3):
            print(f"  Channel {i}: {int(c*255)}")
        print("\n")
        
        
    colnames = ["Image", "Ground Truth", "DinoEnc","UNetDino","UNetDinoAttGate","Vanilla U-Net"]
    rownames = ["Sagittal","Axial","Coronal"]  

    
    for ax, col in zip(axs[0], colnames):
        ax.set_title(col)

    for i, row in enumerate(rownames):
        fig.text(
            0.04,                     
            0.83 - i * 0.33,          
            row,
            va='center',
            ha='center',
            rotation=90,
            fontsize=16
        )



   
    target_size = (256, 256)
    dim_dicts = {
        'dinov3_vits16': 384,
        'dinov3_vits16plus': 384,
        'dinov3_vitb16': 768,
        'dinov3_vitl16': 1024,
        'dinov3_vith16plus': 1080
        }

    modelPaths = {
        'dinov3_vits16': '../DinoWeights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': '../DinoWeights/dinov3_vits16plus.pth',
        'dinov3_vitb16': '../DinoWeights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': '../DinoWeights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vith16plus': '../DinoWeights/dinov3_vith16plusw.pth'
    }

    # check if the weight path exist
    curr_path = modelPaths[args.dinomodel]
    if not os.path.exists(curr_path):
        raise FileNotFoundError(f"Weight file for model {args.dinomodel} not found at path: {curr_path}")
    else:
        print(f"Found weight file for model {args.dinomodel} at path: {curr_path}")



    samp = VolSlicedataset(data_dir, planes[0], [args.Subject])[0]
    testimg = samp["images"][0, :, :, :].to(device)
    embed_dim = dim_dicts[args.dinomodel]
    num_skips = 3
    base_ch = embed_dim
    num_classes = max(GROUP_MAP.values()) + 1

    # layers based on which model
    if args.dinomodel in ['dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16']:
        layers = [2,5,8,11]
    if args.dinomodel in ['dinov3_vitl16','dinov3_vith16plus']:
        layers = [5,11,17,23]

    
    model = DINOv3UNetDecoder(embed_dim=embed_dim, num_skips=num_skips, base_ch=base_ch,
                            num_classes=num_classes, out_size=target_size)

    model.load_state_dict(torch.load(DinoEncPath, map_location="cpu"))
    model = model.to(device)
    model.eval()


    dinov3_model = torch.hub.load("dinov3", args.dinomodel, source="local", weights=modelPaths[args.dinomodel])
    dinov3_model = dinov3_model.to(device)
    rownum = 0
    subject_vol = {}
    with torch.no_grad():
        for plane in planes:
            print(f"Processing plane: {plane}")
            slice_ds = VolSlicedataset(data_dir, plane, [args.Subject])

            selsub = slice_ds.get_subject(args.Subject)
            all_predictions = []
            

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



            axs[rownum, 0].imshow(plotimg, cmap="gray")
            axs[rownum, 0].axis("off")
            # plot gt mask in 0,1
            axs[rownum, 1].imshow(plotimg, cmap="gray")
            axs[rownum, 1].imshow(slice_mask[0, :, :].numpy(), cmap=cmap, alpha=0.7)
            axs[rownum, 1].axis("off")

            axs[rownum, 2].imshow(plotimg, cmap="gray")
            axs[rownum, 2].imshow(pred_mask, cmap=cmap, alpha=0.7)
            axs[rownum, 2].axis("off")

            rownum += 1

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
            del all_predictions, prob_vol
            torch.cuda.empty_cache()
            gc.collect()
    
    # fuse
    path = os.path.join(data_dir, args.Subject)
    image, meta = load_nifti(os.path.join(path, "ct.nii.gz"))
    outshape = image.shape
    p_axial = resample_to_reference(subject_vol["axial"], outshape)
    p_coronal = resample_to_reference(subject_vol["coronal"], outshape)
    p_sagittal = resample_to_reference(subject_vol["sagittal"], outshape)

    fused_prob = fuse_probabilities(p_axial, p_coronal, p_sagittal, weights=(0.33, 0.33, 0.34))
    fused_seg = torch.argmax(fused_prob, dim=0).numpy().astype(np.uint8)

    # save fused segmentation as nifti
    fused_img = nib.Nifti1Image(fused_seg, affine=meta.affine, header=meta.header)
    os.makedirs(os.path.join(save3dpath, args.Subject), exist_ok=True)
    nib.save(fused_img, os.path.join(save3dpath, args.Subject, f"DinoEnc_fused_seg_{args.TrainSizePercent}.nii.gz"))

    

    # delete model 
    del model
    
    embed_dim = dim_dicts[args.dinomodel]*2
    in_channels = testimg.shape[1]
    model = DINOv3UNetEncodeDecoder(
        in_channels=in_channels,
        embed_dim=embed_dim,
        dinov3_model=dinov3_model,
        n_layers=layers,
        num_classes=num_classes,
        out_size=target_size
    )

    model.load_state_dict(torch.load(v1path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    rownum = 0
    subject_vol = {}
    with torch.no_grad():
        for plane in planes:
            slice_ds = VolSlicedataset(data_dir, plane, [args.Subject])

            selsub = slice_ds.get_subject(args.Subject)
            all_predictions = []

            sample_idx = selsub['images'].shape[0] // 2
            slice_img = selsub['images'][sample_idx, :, :, :].to(device)
            slice_mask = selsub['masks'][sample_idx, :, :]

            output = model(slice_img).squeeze(0)
            pred_mask = torch.argmax(output, dim=0).cpu().numpy()

            plotimg = slice_img.squeeze(0).cpu().numpy()
            plotimg = plotimg[0,:,:]


            axs[rownum, 3].imshow(plotimg, cmap="gray")
            axs[rownum, 3].imshow(pred_mask, cmap=cmap, alpha=0.7)
            axs[rownum, 3].axis("off")

            rownum += 1

            for slice_idx in range(selsub['images'].shape[0]):
                slice_img = selsub['images'][slice_idx, :, :, :].to(device)
                output = model(slice_img).squeeze(0)
                all_predictions.append(output.cpu().numpy())

                del slice_img, output
                torch.cuda.empty_cache()

            prob_vol = reconstruct_volume(all_predictions, plane)
            subject_vol[plane] = prob_vol.cpu()
            del all_predictions, prob_vol
            torch.cuda.empty_cache()
            gc.collect()

    # fuse
    path = os.path.join(data_dir, args.Subject)
    image, meta = load_nifti(os.path.join(path, "ct.nii.gz"))
    outshape = image.shape
    p_axial = resample_to_reference(subject_vol["axial"], outshape)
    p_coronal = resample_to_reference(subject_vol["coronal"], outshape)
    p_sagittal = resample_to_reference(subject_vol["sagittal"], outshape)

    fused_prob = fuse_probabilities(p_axial, p_coronal, p_sagittal, weights=(0.33, 0.33, 0.34))
    fused_seg = torch.argmax(fused_prob, dim=0).numpy().astype(np.uint8)

    # save fused segmentation as nifti
    fused_img = nib.Nifti1Image(fused_seg, affine=meta.affine, header=meta.header)
    os.makedirs(os.path.join(save3dpath, args.Subject), exist_ok=True)
    nib.save(fused_img, os.path.join(save3dpath, args.Subject, f"UNetDino_fused_seg_{args.TrainSizePercent}.nii.gz"))


    # delete model
    del model

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
    
    model.load_state_dict(torch.load(AGv2Path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    rownum = 0
    subject_vol = {}
    with torch.no_grad():
        for plane in planes:
            slice_ds = VolSlicedataset(data_dir, plane, [args.Subject])

            selsub = slice_ds.get_subject(args.Subject)
            all_predictions = []

            sample_idx = selsub['images'].shape[0] // 2
            slice_img = selsub['images'][sample_idx, :, :, :].to(device)
            slice_mask = selsub['masks'][sample_idx, :, :]

            output = model(slice_img).squeeze(0)
            pred_mask = torch.argmax(output, dim=0).cpu().numpy()

            plotimg = slice_img.squeeze(0).cpu().numpy()
            plotimg = plotimg[0,:,:]


            axs[rownum, 4].imshow(plotimg, cmap="gray")
            axs[rownum, 4].imshow(pred_mask, cmap=cmap, alpha=0.7)
            axs[rownum, 4].axis("off")

            rownum += 1
            for slice_idx in range(selsub['images'].shape[0]):
                slice_img = selsub['images'][slice_idx, :, :, :].to(device)
                output = model(slice_img).squeeze(0)
                all_predictions.append(output.cpu().numpy())

                del slice_img, output
                torch.cuda.empty_cache()
            
            prob_vol = reconstruct_volume(all_predictions, plane)
            subject_vol[plane] = prob_vol.cpu()
            del all_predictions, prob_vol
            torch.cuda.empty_cache()
            gc.collect()
    # fuse
    path = os.path.join(data_dir, args.Subject)
    image, meta = load_nifti(os.path.join(path, "ct.nii.gz"))
    outshape = image.shape
    p_axial = resample_to_reference(subject_vol["axial"], outshape)
    p_coronal = resample_to_reference(subject_vol["coronal"], outshape)
    p_sagittal = resample_to_reference(subject_vol["sagittal"], outshape)   

    fused_prob = fuse_probabilities(p_axial, p_coronal, p_sagittal, weights=(0.33, 0.33, 0.34))
    fused_seg = torch.argmax(fused_prob, dim=0).numpy().astype(np.uint8)

    # save fused segmentation as nifti
    fused_img = nib.Nifti1Image(fused_seg, affine=meta.affine, header=meta.header)
    os.makedirs(os.path.join(save3dpath, args.Subject), exist_ok=True)
    nib.save(fused_img, os.path.join(save3dpath, args.Subject, f"UNetDinoAGFullV2_fused_seg_{args.TrainSizePercent}.nii.gz"))

    del model, dinov3_model

    model = VanillaUNet(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(torch.load(VanillaUnetPath, map_location="cpu"))
    model = model.to(device)
    model.eval()

    rownum = 0
    subject_vol = {}
    with torch.no_grad():
        for plane in planes:
            slice_ds = VolSlicedataset(data_dir, plane, [args.Subject])

            selsub = slice_ds.get_subject(args.Subject)
            all_predictions = []

            sample_idx = selsub['images'].shape[0] // 2
            slice_img = selsub['images'][sample_idx, :, :, :].to(device)
            slice_mask = selsub['masks'][sample_idx, :, :]

            output = model(slice_img).squeeze(0)
            pred_mask = torch.argmax(output, dim=0).cpu().numpy()

            plotimg = slice_img.squeeze(0).cpu().numpy()
            plotimg = plotimg[0,:,:]


            axs[rownum, 5].imshow(plotimg, cmap="gray")
            axs[rownum, 5].imshow(pred_mask, cmap=cmap, alpha=0.7)
            axs[rownum, 5].axis("off")

            rownum += 1
            for slice_idx in range(selsub['images'].shape[0]):
                slice_img = selsub['images'][slice_idx, :, :, :].to(device)
                output = model(slice_img).squeeze(0)
                all_predictions.append(output.cpu().numpy())

                del slice_img, output
                torch.cuda.empty_cache()

            prob_vol = reconstruct_volume(all_predictions, plane)
            subject_vol[plane] = prob_vol.cpu()
            del all_predictions, prob_vol
            torch.cuda.empty_cache()
            gc.collect()

    # fuse
    path = os.path.join(data_dir, args.Subject)
    image, meta = load_nifti(os.path.join(path, "ct.nii.gz"))
    outshape = image.shape
    p_axial = resample_to_reference(subject_vol["axial"], outshape)
    p_coronal = resample_to_reference(subject_vol["coronal"], outshape)
    p_sagittal = resample_to_reference(subject_vol["sagittal"], outshape)

    fused_prob = fuse_probabilities(p_axial, p_coronal, p_sagittal, weights=(0.33, 0.33, 0.34))
    fused_seg = torch.argmax(fused_prob, dim=0).numpy().astype(np.uint8)

    # save fused segmentation as nifti
    fused_img = nib.Nifti1Image(fused_seg, affine=meta.affine, header=meta.header)
    os.makedirs(os.path.join(save3dpath, args.Subject), exist_ok=True)
    nib.save(fused_img, os.path.join(save3dpath, args.Subject, f"VanillaUNet_fused_seg_{args.TrainSizePercent}.nii.gz"))

    # also make a ground true version
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
    fused_img = nib.Nifti1Image(out, affine=meta.affine, header=meta.header)
    nib.save(fused_img, os.path.join(save3dpath, args.Subject, f"GroundTruth_seg.nii.gz"))


    # save the figure
    plt.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.90,
    bottom=0.05,
    wspace=0.01,   # horizontal spacing
    hspace=0.01    # vertical spacing
    )
    save_dir  = "PredictionsPlots/"+args.Subject
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"prediction_{args.Subject}_trainsize{args.TrainSizePercent}.png"))
    # also in pdf
    plt.savefig(os.path.join(save_dir, f"prediction_{args.Subject}_trainsize{args.TrainSizePercent}.pdf"))
    plt.close()

    # Legend plotting
    legend_fig = plt.figure(figsize=(14, 0.8))




    patches = []
    for label_id in sorted(label_names.keys()):
        if label_id == 0:
            continue  # skip background in legend (recommended)
        #color = cmap(label_id)
        color = blend_with_gray(cmap(label_id), gray=0.5, alpha=0.7)  # blend with gray for better visibility
        patches.append(
            
            mpatches.Patch(color=color, label=label_names[label_id])
        )

    legend_fig.legend(
        handles=patches,
        loc="center",
        ncol=6,              
        frameon=False,
        fontsize=10

    )

    plt.axis("off")

    plt.savefig(
        f"class_legend.png",
        bbox_inches="tight",
        pad_inches=0.01
    )

    plt.close()









if __name__ == "__main__":
    main()