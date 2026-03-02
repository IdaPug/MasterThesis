import os
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import shutil

from DataClasses import *
from OrganLabels import *


GROUP_MAP = {
    0: 0,
    **dict.fromkeys([1,2,3,5,7,8,9], 1),
    **dict.fromkeys([4,6,18,19,20], 2),
    **dict.fromkeys([21,22,23,24], 3),
    **dict.fromkeys(range(10,17), 4),
    17: 4,
    **dict.fromkeys(range(51,69), 5),
    90: 6, 91: 6,
    79: 7,
    **dict.fromkeys(range(25,51), 8),
    **dict.fromkeys(range(92,118), 9),
    **dict.fromkeys(range(69,79), 10),
    **dict.fromkeys(range(80,90), 11),
}

label_names = {
    0: "background",
    1: "abdominal solid organs",
    2: "GI tract",
    3: "urinary_reproductive",
    4: "respiratory system",
    5: "heart & vessels",
    6: "head",
    7: "spinal cord",
    8: "spine",
    9: "rib cages",
    10: "appendicular skeleton",
    11: "muscles",
}

def collate_no_pad(batch):
    return batch[0]

target_size = (256, 256)

slice_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.CenterCrop(target_size)
])

mask_transform = transforms.Lambda(
    lambda mask: F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=target_size,
        mode="nearest"
    ).squeeze(0).squeeze(0).long()
)

parser = argparse.ArgumentParser()
parser.add_argument("--shard_id", type=int,default=1,
                    help="Shard ID (0-indexed)")
parser.add_argument("--num_shards", type=int,default=1,
                    help="Total number of shards")
args = parser.parse_args()

SHARD_ID = args.shard_id-1
NUM_SHARDS = args.num_shards

data_dir = "../TotalSegmentator_v2_Full/TestSubjects_alt/"
data_dir_alt = data_dir



planes = ["sagittal", "axial", "coronal"]

# Load full volume dataset
vol_ds_full = TotalSegmentatorV2Dataset(
    root_dir=data_dir,
    group_map=GROUP_MAP,
)

indices = [
        i for i in range(len(vol_ds_full))
        if i % NUM_SHARDS == SHARD_ID
    ]

print(f"Shard {SHARD_ID}: {len(indices)} subjects")

vol_ds = Subset(vol_ds_full, indices)

datasets = {}

for plane in planes:
    datasets[plane] = VolumeSliceDataset(
        base_dataset=vol_ds,
        plane=plane,
        slice_transform=slice_transform,
        mask_transform=mask_transform,
        slice_step=1
    )


for plane in planes:
    print(f"[Shard {SHARD_ID}] Processing plane: {plane}")

    data  = datasets[plane]

    data_loader = DataLoader(
        data,
        batch_size=1,
        collate_fn=collate_no_pad
    )
    
    
    for i, sample in enumerate(data_loader):
        subject = sample["subject"]
        imgs = sample["image"]   # (num_slices, 3, H, W)
        masks = sample["mask"]   # (num_slices, H, W)

        print(f"[Shard {SHARD_ID}] Subject {subject} ({i+1}/{len(data_loader)})")

        means = []
        stds = []
        for j in range(imgs.shape[0]):
            img2d = imgs[j].unsqueeze(0)
            mask2d = masks[j].unsqueeze(0)

            means.append(img2d.float().mean().item())
            stds.append(img2d.float().std().item())


            filepath = (
                f"{data_dir_alt}/{subject}/ImgSlices/{plane}/slice{j:03d}.pt"
            )

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Skip if already processed (safe restart)
            if os.path.exists(filepath):
                continue

            torch.save(
                {
                    "image": img2d.half().cpu(),
                    "mask": mask2d.cpu().to(torch.uint8),
                },
                filepath
            )

        mean_intensity = np.mean(means)
        std_intensity = np.mean(stds)
        print(f"[Shard {subject}] Plane {plane}: Mean intensity: {mean_intensity:.4f}, Std intensity: {std_intensity:.4f}")
    
    print(f"[Shard {SHARD_ID}] Finished plane {plane}")

print(f"[Shard {SHARD_ID}] DONE")
