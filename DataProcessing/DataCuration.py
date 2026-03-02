import numpy as np
import glob
import os
import math
from collections import defaultdict
import torch
import matplotlib.pyplot as plt

from DataClasses import Slicedataset

NUM_CLASSES = 12
FG_THRESHOLD = 0.1
CE_THRESHOLD = 0.8


    
def Slice_stats(mask):
    # make mask numpy array
    total_pixels = mask.size

    unique, counts = np.unique(mask, return_counts=True)
    area = dict(zip(unique, counts))

    fg_pixels = total_pixels - area.get(0, 0)
    fg_ratio = fg_pixels / total_pixels

    if fg_ratio < FG_THRESHOLD:
        return None

    class_presence = np.zeros(NUM_CLASSES, dtype=np.int32)
    class_areas = np.zeros(NUM_CLASSES, dtype=np.float32)

    for c, cnt in area.items():
        class_presence[c] = 1
        if c != 0:
            class_areas[c] = cnt / fg_pixels
        
    
    entropy = 0.0
    for a in class_areas[1:]:
        if a > 0:
            entropy -= a * math.log2(a)

    if entropy < CE_THRESHOLD:
        return None
    
    num_classes = class_presence[1:].sum()

    return fg_ratio, entropy, num_classes, class_presence


def get_subject_max_indices(ds):
    max_idx = defaultdict(int)
    for i in range(len(ds)):
        print(f"Getting max indices: processing dataset index {i}/{len(ds)}", end="\r")
        s = ds[i]
        max_idx[s["subject"]] = max(max_idx[s["subject"]], s["idx"])
    return max_idx


def collect_candidates(ds):
    #max_idx = get_subject_max_indices(ds)
    candidates = []

    for i in range(len(ds)):
        s = ds[i]
        mask = s["mask"].squeeze().numpy()
        #print(f"Processing subject {s['subject']}, slice {s['idx']}")

        stats = Slice_stats(mask)
        if stats is None:
            continue

        fg_ratio, entropy, num_classes, class_presence = stats
        #print(f"max_idx[s['subject']]: {max_idx[s['subject']]}")
        #slice_pos = s["idx"] / max_idx[s["subject"]]
        #print(f"Slice pos: {slice_pos:.3f}, FG ratio: {fg_ratio:.4f}, Entropy: {entropy:.4f}, Num classes: {num_classes}")

        candidates.append({
            "dataset_idx": i,
            "subject": s["subject"],
            "slice_idx": s["idx"],
            "fg_ratio": fg_ratio,
            "entropy": entropy,
            "num_classes": num_classes,
            "class_presence": class_presence,
        })

    return candidates

def assign_band(pos):
    if pos < 0.2:
        return 0
    elif pos < 0.4:
        return 1
    elif pos < 0.6:
        return 2
    elif pos < 0.8:
        return 3
    else:
        return 4

def select_class_anchors(candidates, min_per_class=3):
    selected = set()
    per_class = defaultdict(list)

    for c in candidates:
        for cls in np.where(c["class_presence"] == 1)[0]:
            if cls > 0:
                per_class[cls].append(c)

    for cls, items in per_class.items():
        items = sorted(
            items,
            key=lambda x: (x["fg_ratio"], x["entropy"]),
            reverse=True
        )
        for it in items[:min_per_class]:
            selected.add(it["dataset_idx"])

    return selected

def score_slice(c):
    return (
        0.5 * c["entropy"]
        + 0.2 * c["fg_ratio"]
        + 0.3 * c["num_classes"]
    )


def select_class_anchors(candidates, min_per_class=3):
    selected = set()
    selected_subjects = set()
    per_class = defaultdict(list)

    for c in candidates:
        for cls in np.where(c["class_presence"] == 1)[0]:
            if cls > 0:
                per_class[cls].append(c)

    for cls, items in per_class.items():
        items = sorted(
            items,
            key=lambda x: (x["fg_ratio"], x["entropy"]),
            reverse=True
        )

        count = 0
        for it in items:
            if it["subject"] in selected_subjects:
                continue
            selected.add(it["dataset_idx"])
            selected_subjects.add(it["subject"])
            count += 1
            if count >= min_per_class:
                break

    return selected

def select_slices(candidates, total=10, min_per_class=3):
    selected = []
    selected_subjects = set()

    # Step 1: class anchors
    anchors = select_class_anchors(candidates, min_per_class)
    for idx in anchors:
        c = next(c for c in candidates if c["dataset_idx"] == idx)
        selected.append(idx)
        selected_subjects.add(c["subject"])

        if len(selected) >= total:
            return selected

    # Step 2: fill remaining slots by score
    scored = sorted(candidates, key=score_slice, reverse=True)
    for c in scored:
        if len(selected) >= total:
            break
        if c["dataset_idx"] in selected:
            continue
        if c["subject"] in selected_subjects:
            continue  

        selected.append(c["dataset_idx"])
        selected_subjects.add(c["subject"])

    return selected

def select_slices_random(candidates, total=10, min_per_class=2):
    selected = set()
    selected_subjects = set()

    # Step 1: class anchors
    anchors = select_class_anchors(candidates, min_per_class)
    for idx in anchors:
        c = next(c for c in candidates if c["dataset_idx"] == idx)
        selected.add(idx)
        selected_subjects.add(c["subject"])

        if len(selected) >= total:
            return list(selected)

    # Step 2: fill remaining slots randomly
    remaining_candidates = [
        c for c in candidates
        if c["dataset_idx"] not in selected and c["subject"] not in selected_subjects
    ]
    np.random.shuffle(remaining_candidates)

    for c in remaining_candidates:
        if len(selected) >= total:
            break
        selected.add(c["dataset_idx"])
        selected_subjects.add(c["subject"])

    return list(selected)

def main():
    num_slice_total = 90
    num_slices_per_plane = int(num_slice_total // 3)

    data_dir = "../TotalSegmentator_v2_Full/TrainSubjects/"
    slice_view_dir = f"../CandidateSlicesPlots{num_slice_total}New/"
    slice_save_dir = f"../CandidateSliceData{num_slice_total}New/"

    # make dirs. Delete if exist and recreate
    if os.path.exists(slice_view_dir):
        import shutil
        shutil.rmtree(slice_view_dir)

    if os.path.exists(slice_save_dir):
        import shutil
        shutil.rmtree(slice_save_dir) 
    
    os.makedirs(slice_view_dir, exist_ok=True)
    os.makedirs(slice_save_dir, exist_ok=True)


    available_subjects = [d for d in glob.glob(os.path.join(data_dir, "s*")) if os.path.isdir(d)]
    subjects_names = [os.path.basename(d) for d in available_subjects]
    print(f"Found {len(subjects_names)} subjects.")

    torch.manual_seed(42)
    np.random.seed(42)
    # get 10 random subjects
    #subjects_names = np.random.choice(subjects_names, size=10, replace=False).tolist()
    #print(f"Selected {len(subjects_names)} subjects for candidate collection.")

    planes = ["sagittal", "axial", "coronal"]

    candidates_train = {}
    candidates_val = {}

    for plane in planes:
        ds = Slicedataset(
            data_dir,
            plane,
            subjects_names
        )
        # get 1% of the slices in the dataset
        num_slices = len(ds)
        num_sampled = 0.1 * num_slices
        sampled_indices = np.random.choice(num_slices, size=int(num_sampled), replace=False)
        ds = torch.utils.data.Subset(ds, sampled_indices)
        print(f"Collecting candidates for plane {plane} from {len(ds)} slices.")
        


        plane_candidates = collect_candidates(ds)
        # print how many candidates were collected vs total slices
        print(f"Collected {len(plane_candidates)} candidates for plane {plane} out of {len(ds)} total slices.")
        

        #selected = select_slices(plane_candidates, total=num_slices_per_plane, min_per_class=5)
        selected = select_slices_random(plane_candidates, total=num_slices_per_plane, min_per_class=2)
        print(f"Selected {len(selected)} slices for plane {plane}. Stats:")


        train_ids = np.random.choice(selected, size=num_slices_per_plane, replace=False).tolist()
        val_ids = [s for s in selected if s not in train_ids]
        candidates_train[plane] = train_ids
        candidates_val[plane] = val_ids

        
        # plot and save the selected slices
        plot_view_dir = os.path.join(slice_view_dir, "train", plane)
        os.makedirs(plot_view_dir, exist_ok=True)
        for idx in train_ids:
            s = ds[idx]
            img = s["image"].squeeze().numpy()
            mask = s["mask"].squeeze().numpy()

            
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(img[0,:,:], cmap="gray")
            ax[0].set_title("Image")
            ax[1].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES-1)
            ax[1].set_title("Mask")
            plt.suptitle(f"Subject: {s['subject']}, Slice: {s['idx']}")
            plt.savefig(os.path.join(plot_view_dir, f"{s['subject']}_slice{s['idx']}.png"))
            plt.close(fig)

            os.makedirs(os.path.join(slice_save_dir, "train", plane), exist_ok=True)
            # also save the slice data as .pt file
            slice_data = {
                "image": s["image"],
                "mask": s["mask"],
                "subject": s["subject"],        
                "plane": plane,
                "idx": s["idx"],
            }
            torch.save(slice_data, os.path.join(slice_save_dir, "train", plane, f"{s['subject']}_slice{s['idx']}.pt"))

        
        plot_view_dir = os.path.join(slice_view_dir, "val", plane)
        os.makedirs(plot_view_dir, exist_ok=True)
        for idx in val_ids:
            s = ds[idx]
            img = s["image"].squeeze().numpy()
            mask = s["mask"].squeeze().numpy()

            
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(img[0,:,:], cmap="gray")
            ax[0].set_title("Image")
            ax[1].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES-1)
            ax[1].set_title("Mask")
            plt.suptitle(f"Subject: {s['subject']}, Slice: {s['idx']}")
            plt.savefig(os.path.join(plot_view_dir, f"{s['subject']}_slice{s['idx']}.png"))
            plt.close(fig)

            os.makedirs(os.path.join(slice_save_dir, "val", plane), exist_ok=True)
            # also save the slice data as .pt file
            slice_data = {
                "image": s["image"],
                "mask": s["mask"],
                "subject": s["subject"],        
                "plane": plane,
                "idx": s["idx"],
            }
            torch.save(slice_data, os.path.join(slice_save_dir, "val", plane,f"{s['subject']}_slice{s['idx']}.pt"))
        
        print(f"Selected {len(selected)} slices for plane {plane}.")
        
    
        


        


        









if __name__ == "__main__":
    main()

