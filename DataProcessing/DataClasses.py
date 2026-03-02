import os
import glob
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from torchvision import tv_tensors
from OrganLabels import OFFICIAL_LABELS


@dataclass
class NiftiMeta:
    affine: np.ndarray                         # 4×4 affine matrix (voxel→world)
    header: nib.nifti1.Nifti1Header            # NIfTI header (contains spacing, datatype, etc.)
    voxel_spacing: tuple[float, float, float]  # (dx, dy, dz)
    orientation: tuple[str, str, str]          # ('R','A','S') etc.
    shape: tuple[int, int, int] 


class TotalSegmentatorV2Dataset(Dataset):
    """
    PyTorch Dataset for TotalSegmentator v2 that:
      * Loads the CT volume (ct.nii.gz)
      * Automatically loads all available organ masks
      * Remaps original labels to super-classes using GROUP_MAP
      * Stores affine, voxel spacing, and orientation metadata for alignment
    """
    
    def __init__(
        self,
        root_dir: str,
        group_map: Optional[Dict[int, int]] = None,
        transforms: Optional[Callable] = None,
        image_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
    ):
        self.root_dir = root_dir
        self.group_map = group_map
        self.transforms = transforms
        self.image_dtype = image_dtype
        self.label_dtype = label_dtype

        # Collect all subject folders containing ct.nii.gz
        self.subjects = sorted(
            [d for d in glob.glob(os.path.join(root_dir, "s*")) if os.path.isdir(d)]
        )

        print(f"Found {len(self.subjects)} subjects in {root_dir}")

    def __len__(self):
        return len(self.subjects)


    def _load_nifti(self, path: str) -> tuple[np.ndarray, NiftiMeta]:
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


    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply GROUP_MAP to convert original labels to super-class labels"""
        if self.group_map is None:
            return mask
        out = np.zeros_like(mask, dtype=np.int16)
        for old_id, new_id in self.group_map.items():
            out[mask == old_id] = new_id
        return out


    def __getitem__(self, idx: int):
        subj_dir = self.subjects[idx]
        ct_path = os.path.join(subj_dir, "ct.nii.gz")
        seg_dir = os.path.join(subj_dir, "segmentations")

        # Load CT
        image, image_meta = self._load_nifti(ct_path)
        image = image.astype(np.float32)

        #  Load all available organ masks
        mask = np.zeros_like(image, dtype=np.int16)
        organ_files = glob.glob(os.path.join(seg_dir, "*.nii.gz"))
        for organ_path in organ_files:
            filename = os.path.basename(organ_path)

            # strip extension(s)
            organ_name = os.path.splitext(os.path.splitext(filename)[0])[0]

            # look up official label id
            try:
                label_id = OFFICIAL_LABELS[organ_name]
            except KeyError:
                print(f"subject {subj_dir} has unknown organ {organ_name}")
                raise ValueError(f"Unknown organ name {organ_name} - add to OFFICIAL_LABELS")

            organ_mask, _ = self._load_nifti(organ_path)
            mask[organ_mask > 0] = label_id

        #  Remap labels if GROUP_MAP is provided 
        mask = self._remap_mask(mask)

        #  Optional transforms
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        # Convert to torch tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.image_dtype)
        mask_tensor = torch.from_numpy(mask).to(self.label_dtype)

        return {
            "image": image_tensor,     # (1, D, H, W)
            "mask": mask_tensor,       # (D, H, W)
            "subject": os.path.basename(subj_dir),
            "meta": image_meta,        # contains affine, voxel spacing, orientation, etc.
        }

class FilteredTotalSegmentatorV2Dataset(Dataset):
    """
    PyTorch Dataset for TotalSegmentator v2 that supports filtering by subject IDs.

    Features:
      * Loads CT volume (ct.nii.gz)
      * Automatically loads all organ masks
      * Optionally filters by a list of subjects (names or paths)
      * Remaps labels using GROUP_MAP
      * Returns tensors with metadata
    """

    def __init__(
        self,
        root_dir: str,
        group_map: Optional[Dict[int, int]] = None,
        transforms: Optional[Callable] = None,
        image_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
        subjects: Optional[List[str]] = None,   
    ):
        self.root_dir = root_dir
        self.group_map = group_map
        self.transforms = transforms
        self.image_dtype = image_dtype
        self.label_dtype = label_dtype

        #  Collect all subject folders 
        all_subjects = sorted(
            [d for d in glob.glob(os.path.join(root_dir, "s*")) if os.path.isdir(d)]
        )

        # Optional filtering 
        if subjects is not None:
            # Normalize user-provided list 
            subjects = set(os.path.basename(s) for s in subjects)
            filtered_subjects = [d for d in all_subjects if os.path.basename(d) in subjects]
            if len(filtered_subjects) == 0:
                raise ValueError(f"No matching subjects found for IDs: {subjects}")
            self.subjects = filtered_subjects
            print(f"Filtered to {len(self.subjects)} / {len(all_subjects)} subjects.")
        else:
            self.subjects = all_subjects
            print(f"Found {len(self.subjects)} subjects in {root_dir}")

    def __len__(self):
        return len(self.subjects)

    def _load_nifti(self, path: str) -> tuple[np.ndarray, 'NiftiMeta']:
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

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply GROUP_MAP to convert original labels to super-class labels."""
        if self.group_map is None:
            return mask
        out = np.zeros_like(mask, dtype=np.int16)
        for old_id, new_id in self.group_map.items():
            out[mask == old_id] = new_id
        return out

    def __getitem__(self, idx: int):
        subj_dir = self.subjects[idx]
        ct_path = os.path.join(subj_dir, "ct.nii.gz")
        seg_dir = os.path.join(subj_dir, "segmentations")

        # ---- Load CT volume ----
        image, image_meta = self._load_nifti(ct_path)
        image = image.astype(np.float32)

        # ---- Load all available organ masks ----
        mask = np.zeros_like(image, dtype=np.int16)
        organ_files = glob.glob(os.path.join(seg_dir, "*.nii.gz"))
        for organ_path in organ_files:
            filename = os.path.basename(organ_path)
            organ_name = os.path.splitext(os.path.splitext(filename)[0])[0]

            # Look up label ID
            try:
                label_id = OFFICIAL_LABELS[organ_name]
            except KeyError:
                raise ValueError(f"Unknown organ '{organ_name}' in {subj_dir}. "
                                 "Add it to OFFICIAL_LABELS.")

            organ_mask, _ = self._load_nifti(organ_path)
            mask[organ_mask > 0] = label_id

        # Remap labels if GROUP_MAP provided
        mask = self._remap_mask(mask)

        # Optional transforms 
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        #  Convert to torch tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.image_dtype)
        mask_tensor = torch.from_numpy(mask).to(self.label_dtype)

        return {
            "image": image_tensor,     # (1, D, H, W)
            "mask": mask_tensor,       # (D, H, W)
            "subject": os.path.basename(subj_dir),
            "meta": image_meta,
        }

class TotalSegmentatorV2SubsetDataset(TotalSegmentatorV2Dataset):
    """
    PyTorch Dataset for TotalSegmentator v2 with an optional subject subset.
    """

    def __init__(
        self,
        root_dir: str,
        subject_list: Optional[list[str]] = None,
        group_map: Optional[Dict[int, int]] = None,
        transforms: Optional[Callable] = None,
        image_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
    ):
        self.root_dir = root_dir
        self.group_map = group_map
        self.transforms = transforms
        self.image_dtype = image_dtype
        self.label_dtype = label_dtype

        # Collect all subject folders
        all_subjects = sorted(
            [d for d in glob.glob(os.path.join(root_dir, "s*")) if os.path.isdir(d)]
        )

        # Filter subjects if a list is provided
        if subject_list is not None:
            self.subjects = [
                d for d in all_subjects if os.path.basename(d) in subject_list
            ]
        else:
            self.subjects = all_subjects

        print(f"Found {len(self.subjects)} subjects in {root_dir}")


class SliceDataset(Dataset):
    """
    Iterate through all slices of all subjects.

    Returns:
        {
          "image":  (3, H, W) tensor  (ready for DINOv2 / torchvision transforms)
          "mask":   (H, W)   tensor   (integer labels)
          "subject": subject id (str)
          "slice_idx": index of the slice in the original volume
        }
    """

    def __init__(self,
                 base_dataset: TotalSegmentatorV2Dataset,
                 plane: str = "axial",
                 slice_transform=None,
                 mask_transform=None):
        """
        Args:
            base_dataset: an instance of TotalSegmentatorV2Dataset
            plane: 'axial', 'sagittal', or 'coronal'
            slice_transform: torchvision transforms for image slices
            mask_transform: optional transform for mask slices
        """
        self.base = base_dataset
        self.plane = plane
        self.slice_transform = slice_transform
        self.mask_transform = mask_transform

        # Precompute (subject_idx, slice_idx) pairs
        self.index = []
        for subj_idx in range(len(base_dataset)):
            img_shape = nib.load(
                base_dataset.subjects[subj_idx] + "/ct.nii.gz"
            ).shape  # (D,H,W)
            if plane == "axial":
                num_slices = img_shape[0]
            elif plane == "coronal":
                num_slices = img_shape[1]
            else:
                num_slices = img_shape[2]
            self.index.extend([(subj_idx, s) for s in range(num_slices)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        subj_idx, slice_idx = self.index[idx]
        sample = self.base[subj_idx]  # loads full volume tensors
        img3d = sample["image"].squeeze(0)  # (D,H,W)
        mask3d = sample["mask"]             # (D,H,W)

        # Select slice along chosen plane
        if self.plane == "axial":
            img2d = img3d[slice_idx, :, :]
            mask2d = mask3d[slice_idx, :, :]
        elif self.plane == "coronal":
            img2d = img3d[:, slice_idx, :]
            mask2d = mask3d[:, slice_idx, :]
        else:  # sagittal
            img2d = img3d[:, :, slice_idx]
            mask2d = mask3d[:, :, slice_idx]

        # normalize to 0-1
        img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-5)

        # Duplicate channel to 3 for DINOv2 (expects 3 RGB channels)
        img2d = img2d.unsqueeze(0).repeat(3, 1, 1)  # (3,H,W)

        if self.slice_transform:
            img2d = self.slice_transform(img2d)
        if self.mask_transform:
            mask2d = self.mask_transform(mask2d)

        return {
            "image": img2d,
            "mask": mask2d,
            "subject": sample["subject"],
            "slice_idx": slice_idx
        }

class CachedSliceDataset(Dataset):
    """
    Fully shuffled 2D slice dataset with on-the-fly volume caching.
    Loads each subject only once into RAM during iteration.
    """

    def __init__(
        self,
        base_dataset,
        plane="axial",
        slice_transform=None,
        mask_transform=None,
        cache_size=2,
    ):
        self.base = base_dataset
        self.plane = plane
        self.slice_transform = slice_transform
        self.mask_transform = mask_transform
        self.cache_size = cache_size
        self._cache = {}

        # Precompute (subject_idx, slice_idx) pairs 
        self.index = []
        for subj_idx, subj_dir in enumerate(base_dataset.subjects):
            ct_path = subj_dir + "/ct.nii.gz"
            shape = nib.load(ct_path).shape  
            if plane == "axial":
                num_slices = shape[0]
            elif plane == "coronal":
                num_slices = shape[1]
            else:
                num_slices = shape[2]
            self.index.extend([(subj_idx, s) for s in range(num_slices)])

    def __len__(self):
        return len(self.index)

    def _get_subject(self, subj_idx):
        # Check if cached
        if subj_idx in self._cache:
            return self._cache[subj_idx]

        # Otherwise load from disk
        sample = self.base[subj_idx]

        # Maintain small cache (FIFO)
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[subj_idx] = sample

        return sample

    def __getitem__(self, idx):
        subj_idx, slice_idx = self.index[idx]
        sample = self._get_subject(subj_idx)

        img3d = sample["image"].squeeze(0)
        mask3d = sample["mask"]

        # Select slice along chosen plane
        if self.plane == "axial":
            img2d, mask2d = img3d[slice_idx], mask3d[slice_idx]
        elif self.plane == "coronal":
            img2d, mask2d = img3d[:, slice_idx, :], mask3d[:, slice_idx, :]
        else:
            img2d, mask2d = img3d[:, :, slice_idx], mask3d[:, :, slice_idx]

        # Normalize to 0–1 range
        img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-5)
        img2d = img2d.unsqueeze(0).repeat(3, 1, 1)  # 3-channel

        # Apply transforms if any
        if self.slice_transform:
            img2d = self.slice_transform(img2d)
        if self.mask_transform:
            mask2d = self.mask_transform(mask2d)

        return {
            "image": img2d,
            "mask": mask2d,
            "subject": sample["subject"],
            "slice_idx": slice_idx,
        }


class VolumeSliceDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        plane: str = "axial",
        slice_transform=None,
        mask_transform=None,
        slice_step: int = 1,      
    ):
        """
        Args:
            base_dataset: TotalSegmentatorV2Dataset (3D volumes)
            plane: 'axial' | 'coronal' | 'sagittal'
            slice_transform: optional transform for each 2D slice
            mask_transform: optional transform for each 2D mask slice
            slice_step: keep every n-th slice (default 1 = no skipping)
        """
        self.base = base_dataset
        self.plane = plane
        self.slice_transform = slice_transform
        self.mask_transform = mask_transform
        self.slice_step = max(1, int(slice_step))  

    def __len__(self):
        # still one entry per subject
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        img3d = sample["image"].squeeze(0)   # (D,H,W)
        mask3d = sample["mask"]              # (D,H,W)

        # reorient based on plane
        if self.plane == "axial":
            imgs, masks = img3d, mask3d                  # (D,H,W)
        elif self.plane == "coronal":
            imgs, masks = img3d.permute(1, 0, 2), mask3d.permute(1, 0, 2)  # (H,D,W)
        elif self.plane == "sagittal":
            imgs, masks = img3d.permute(2, 0, 1), mask3d.permute(2, 0, 1)  # (W,D,H)
        else:
            raise ValueError(f"Unknown plane '{self.plane}'")

        # --- NEW: slice spacing ---
        if self.slice_step > 1:
            imgs = imgs[::self.slice_step]
            masks = masks[::self.slice_step]

        # # normalize to [0,1] and duplicate channels
        # imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-5)
        # imgs = imgs.unsqueeze(1).repeat(1, 3, 1, 1)  # (num_slices, 3, H, W)

        # Normalize using ImageNet stats and convert to 3 channels
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]



        # Soft tissue window (TotalSegmentator default)
        WL, WW = 40, 400
        low, high = WL - WW/2, WL + WW/2

        imgs = imgs.clamp(low, high)
        imgs = (imgs - low) / (high - low)

        imgs = imgs.unsqueeze(1).repeat(1, 3, 1, 1)  # (num_slices, 3, H, W)
        # imagenet normalization
        for c in range(3):
            # print mean and std before normalization
            #print(f"Slice: Image mean: {imgs[:, c, :, :].mean().item()}, std: {imgs[:, c, :, :].std().item()}")
            imgs[:, c, :, :] = (imgs[:, c, :, :] - mean[c]) / std[c]
            
    

        # optional transforms
        if self.slice_transform:
            imgs = torch.stack([self.slice_transform(i) for i in imgs])
        if self.mask_transform:
            masks = torch.stack([self.mask_transform(m) for m in masks])

        return {
            "image": imgs,            # (num_slices, 3, H, W)
            "mask": masks,            # (num_slices, H, W)
            "subject": sample["subject"],
        }


class FilteredVolumeSliceDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        subject_ids=None,
        plane: str = "axial",
        slice_transform=None,
        mask_transform=None,
        slice_step: int = 1,
    ):
        """
        A standalone 2D slice dataset that optionally filters by subject IDs.

        Args:
            base_dataset: Base dataset with 3D volumes (e.g., TotalSegmentatorV2Dataset)
            subject_ids: Optional list of subject identifiers to include
            plane: 'axial' | 'coronal' | 'sagittal'
            slice_transform: optional transform for each 2D image slice
            mask_transform: optional transform for each 2D mask slice
            slice_step: keep every n-th slice (default 1 = all slices)
        """
        self.base = base_dataset
        self.plane = plane
        self.slice_transform = slice_transform
        self.mask_transform = mask_transform
        self.slice_step = max(1, int(slice_step))  

        # Apply filtering by subject IDs
        if subject_ids is not None:
            print(f"Filtering FilteredVolumeSliceDataset to {len(subject_ids)} subjects")
            # Find indices matching given subject IDs
            self.filtered_indices = []
            for subj_idx in range(len(base_dataset)):
                sample = base_dataset[subj_idx]
                print("Checking subject:", sample["subject"])
                if sample["subject"] in subject_ids:
                    self.filtered_indices.append(subj_idx)

            if len(self.filtered_indices) == 0:
                raise ValueError("No subjects found matching the given subject_ids.")
        else:
            # No filtering — include all subjects
            self.filtered_indices = list(range(len(base_dataset)))

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Map filtered index to base dataset index
        base_idx = self.filtered_indices[idx]
        sample = self.base[base_idx]

        img3d = sample["image"].squeeze(0)  # (D,H,W)
        mask3d = sample["mask"]             # (D,H,W)

        # Reorient based on plane 
        if self.plane == "axial":
            imgs, masks = img3d, mask3d
        elif self.plane == "coronal":
            imgs, masks = img3d.permute(1, 0, 2), mask3d.permute(1, 0, 2)
        elif self.plane == "sagittal":
            imgs, masks = img3d.permute(2, 0, 1), mask3d.permute(2, 0, 1)
        else:
            raise ValueError(f"Unknown plane '{self.plane}'")

        # Apply slice step 
        if self.slice_step > 1:
            imgs = imgs[::self.slice_step]
            masks = masks[::self.slice_step]

        #  Normalize & expand channels 
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-5)
        imgs = imgs.unsqueeze(1).repeat(1, 3, 1, 1)  # (num_slices, 3, H, W)

        # Apply transforms 
        if self.slice_transform:
            imgs = torch.stack([self.slice_transform(i) for i in imgs])
        if self.mask_transform:
            masks = torch.stack([self.mask_transform(m) for m in masks])

        return {
            "image": imgs,           # (num_slices, 3, H, W)
            "mask": masks,           # (num_slices, H, W)
            "subject": sample["subject"],
        }

class DINOFFeatureDataset(Dataset):
    def __init__(self, root_dir: str, plane: str = "axial"):
        self.root_dir = root_dir
        self.plane = plane

        # collect all feature files
        self.files = sorted(glob.glob(os.path.join(root_dir, "s*", "DINO_Features", plane, "slice*.pt")))
        print(f"Found {len(self.files)} feature files in {root_dir}")
        if len(self.files) == 0:
            raise ValueError("No feature files found - check root_dir and plane")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        features = data["features"].float()   # (num_patches, feature_dim)
        mask = data["mask"].long()             # (H, W)
        meta = data["meta"]                    # dict with patient_id, slice_idx, plane

        return {
            "features": features,
            "mask": mask,
            "meta": meta
        }   
    
class DinoSubjectDataset(Dataset):
    """
    Returns all slices (features + masks) for one subject.
    Each item:
        features: FloatTensor [N, C, H', W']
        masks:    LongTensor  [N, H, W]
        meta_list: list of dicts (one per slice)
    """
    def __init__(self, root_dir, slice_plane="axial"):
        self.root_dir = root_dir
        self.slice_plane = slice_plane

        # gather subject folders starting with 's'
        self.subjects = []
        subject_dirs = sorted(glob.glob(os.path.join(root_dir, "s*")))
        for subj_dir in subject_dirs:
            slice_files = sorted(
                glob.glob(os.path.join(subj_dir, "DINO_Features", slice_plane, "slice*.pt"))
            )
            if slice_files:
                subject_id = os.path.basename(subj_dir)
                self.subjects.append((subject_id, slice_files))

        if len(self.subjects) == 0:
            raise ValueError(f"No valid subject directories found in {root_dir} for plane {slice_plane}")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id, slice_files = self.subjects[idx]

        features_list, masks_list, meta_list = [], [], []
        for f in slice_files:
            d = torch.load(f)
            # d["features"]: [C, H', W']
            features_list.append(d["features"].unsqueeze(0))  # [1, C, H', W']
            masks_list.append(d["mask"].unsqueeze(0))         # [1, H, W]
            meta_list.append(d["meta"])

        features = torch.cat(features_list, dim=0)  # [N, C, H', W']
        masks    = torch.cat(masks_list, dim=0)     # [N, H, W]

        return {
            "subject_id": subject_id,
            "features": features.float(),  
            "masks": masks.long(),
            "meta_list": meta_list
        }


class FilteredDINOFFeatureDataset(Dataset):
    """
    Slice-level dataset for DINO features.
    Optionally restricts loading to specific subject IDs.
    """
    def __init__(self, root_dir: str, plane: str = "axial", subject_ids: list[str] = None):
        """
        Args:
            root_dir (str): Root directory containing subject folders.
            plane (str): Slice plane ("axial", "sagittal", "coronal").
            subject_ids (list[str], optional): List of subject IDs to include
                (e.g., ["s001", "s002"]). If None, includes all subjects.
        """
        self.root_dir = root_dir
        self.plane = plane
        self.subject_ids = subject_ids

        # If subject_ids given, build file list only from those subjects
        if subject_ids is not None:
            self.files = []
            for sid in subject_ids:
                subj_pattern = os.path.join(root_dir, sid, "DINO_Features", plane, "slice*.pt")
                subj_files = sorted(glob.glob(subj_pattern))
                if len(subj_files) == 0:
                    print(f"⚠️ Warning: No feature files found for subject {sid} ({subj_pattern})")
                self.files.extend(subj_files)
        else:
            # Otherwise load all available subjects
            self.files = sorted(glob.glob(os.path.join(root_dir, "s*", "DINO_Features", plane, "slice*.pt")))

        if len(self.files) == 0:
            raise ValueError(f"No feature files found in {root_dir} for plane '{plane}' and given subject_ids={subject_ids}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        features = data["features"].float()   # (num_patches, feature_dim)
        mask = data["mask"].long()             # (H, W)
        meta = data["meta"]                    # dict with patient_id, slice_idx, plane

        return {
            "features": features,
            "mask": mask,
            "meta": meta,
        }

class DinoSubjectCLSDataset(Dataset):
    """
    Returns all CLS tokens for one subject.
    Each item:
        cls_tokens: FloatTensor [N, C]
        meta_list: list of dicts (one per slice)

    """
    def __init__(self, root_dir, slice_plane="axial"):
        self.root_dir = root_dir
        self.slice_plane = slice_plane

        # gather subject folders starting with 's'
        self.subjects = []
        subject_dirs = sorted(glob.glob(os.path.join(root_dir, "s*")))
        for subj_dir in subject_dirs:
            slice_files = sorted(
                glob.glob(os.path.join(subj_dir, "DINO_Features", slice_plane, "slice*.pt"))
            )
            if slice_files:
                subject_id = os.path.basename(subj_dir)
                self.subjects.append((subject_id, slice_files))

        if len(self.subjects) == 0:
            raise ValueError(f"No valid subject directories found in {root_dir} for plane {slice_plane}")
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject_id, slice_files = self.subjects[idx]
        cls_list, meta_list = [], []

        for f in slice_files:
            d = torch.load(f)
          
            cls_list.append(d["cls_token"].unsqueeze(0))  # [1, C]
            meta_list.append(d["meta"])

        cls_tokens = torch.cat(cls_list, dim=0)  # [N, C]
        return {
            "subject_id": subject_id,
            "cls_tokens": cls_tokens.float(),
            "meta_list": meta_list
        }


class FilteredDinoSubjectDataset(Dataset):
    """
    A filtered wrapper around DinoSubjectDataset that only includes
    selected subjects (by subject IDs).

    Example:
        base_ds = DinoSubjectDataset("/data/totalseg_dino", slice_plane="axial")
        filtered_ds = FilteredDinoSubjectDataset(base_ds, subject_ids=["s001", "s005"])
    """
    
    def __init__(
        self,
        base_dataset: DinoSubjectDataset,
        subject_ids: Optional[List[str]] = None,
    ):
        self.base_dataset = base_dataset

        # If no filter given, use all subjects
        if subject_ids is None:
            self.subjects = base_dataset.subjects
        else:
            available_ids = {sid for sid, _ in base_dataset.subjects}
            missing = [sid for sid in subject_ids if sid not in available_ids]
            if missing:
                raise ValueError(f"Subject IDs not found in dataset: {missing}")

            # Keep only the selected ones
            self.subjects = [
                (sid, slice_files)
                for sid, slice_files in base_dataset.subjects
                if sid in subject_ids
            ]

        print(f"FilteredDinoSubjectDataset: {len(self.subjects)} / {len(base_dataset.subjects)} subjects retained")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id, slice_files = self.subjects[idx]

        features_list, masks_list, meta_list = [], [], []
        for f in slice_files:
            d = torch.load(f)
            features_list.append(d["features"].unsqueeze(0))  # [1, C, H', W']
            masks_list.append(d["mask"].unsqueeze(0))         # [1, H, W]
            meta_list.append(d["meta"])

        features = torch.cat(features_list, dim=0)  # [N, C, H', W']
        masks = torch.cat(masks_list, dim=0)        # [N, H, W]

        return {
            "subject_id": subject_id,
            "features": features.float(),
            "masks": masks.long(),
            "meta_list": meta_list,
        }
    
class FilteredDinoSubjectDatasetOrdered(Dataset):
    """
    A filtered wrapper around DinoSubjectDataset that only includes
    selected subjects (by subject IDs), preserving the input order.

    Example:
        base_ds = DinoSubjectDataset("/data/totalseg_dino", slice_plane="axial")
        filtered_ds = FilteredDinoSubjectDataset(base_ds, subject_ids=["s001", "s005"])
    """

    def __init__(
        self,
        base_dataset: DinoSubjectDataset,
        subject_ids: Optional[List[str]] = None,
    ):
        self.base_dataset = base_dataset

        # If no filter given, use all subjects
        if subject_ids is None:
            self.subjects = base_dataset.subjects
        else:
            # Create a lookup dict for quick access
            subj_dict = {sid: slice_files for sid, slice_files in base_dataset.subjects}

            # Check for missing IDs
            missing = [sid for sid in subject_ids if sid not in subj_dict]
            if missing:
                raise ValueError(f"Subject IDs not found in dataset: {missing}")

            # Preserve input order
            self.subjects = [(sid, subj_dict[sid]) for sid in subject_ids]

        print(
            f"FilteredDinoSubjectDataset: {len(self.subjects)} / {len(base_dataset.subjects)} subjects retained"
        )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id, slice_files = self.subjects[idx]

        features_list, masks_list, meta_list = [], [], []
        for f in slice_files:
            d = torch.load(f)
            features_list.append(d["features"].unsqueeze(0))  # [1, C, H', W']
            masks_list.append(d["mask"].unsqueeze(0))         # [1, H, W]
            meta_list.append(d["meta"])

        features = torch.cat(features_list, dim=0)  # [N, C, H', W']
        masks = torch.cat(masks_list, dim=0)        # [N, H, W]

        return {
            "subject_id": subject_id,
            "features": features.float(),
            "masks": masks.long(),
            "meta_list": meta_list,
        }



class PCATransformedDataset(Dataset):
    """Wraps a dataset and applies a pre-trained PCA model to each feature tensor."""
    def __init__(self, base_dataset, pca_model):
        self.base_dataset = base_dataset
        self.pca = pca_model

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        features = sample["features"]  # [C, H, W]
        C, H, W = features.shape

        # Flatten spatial dims for PCA
        features_flat = features.reshape(C, -1).T  # [H*W, C]
        reduced_flat = self.pca.transform(features_flat)  # NumPy float64 [H*W, n_components]

        # Convert to torch.float32
        reduced = torch.tensor(reduced_flat.T, dtype=torch.float32).reshape(self.pca.n_components_, H, W)

        # Replace features with reduced version
        sample["features"] = reduced
        return sample


class PCATransformSubjectDataset(Dataset):
    """
    Wraps a FilteredDinoSubjectDataset to apply a pre-trained PCA
    along the channel dimension for every slice in each subject.
    """

    def __init__(self, base_dataset, pca_model):
        self.base_dataset = base_dataset
        self.pca = pca_model

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]  # dict with keys: subject_id, features, masks, meta_list
        features = sample["features"]    # [N, C, H, W]
        N, C, H, W = features.shape

        reduced_slices = []
        for i in range(N):
            currslice = features[i]                  # [C, H, W]
            currslice_flat = currslice.reshape(C, -1).T  # [H*W, C]
            reduced_flat = self.pca.transform(currslice_flat)  # [H*W, C']
            reduced = torch.tensor(reduced_flat.T, dtype=torch.float32).reshape(
                self.pca.n_components_, H, W
            )
            reduced_slices.append(reduced.unsqueeze(0))  # [1, C', H, W]

        reduced_features = torch.cat(reduced_slices, dim=0)  # [N, C', H, W]

        return {
            "subject_id": sample["subject_id"],
            "features": reduced_features,
            "masks": sample["masks"],
            "meta_list": sample["meta_list"],
        }
    

class FilteredDINOFeatureAndSkipDataset(Dataset):
    """
    Slice-level dataset that loads both DINO features and DINO skip features.
    Matches each slice's main feature file with its corresponding skip feature file.
    """
    def __init__(self, root_dir: str, plane: str = "axial", subject_ids: list[str] = None):
        """
        Args:
            root_dir (str): Root directory containing subject folders.
            plane (str): Slice plane ("axial", "sagittal", "coronal").
        subject_ids (list[str], optional): List of subject IDs to include.
    """
        self.root_dir = root_dir
        self.plane = plane
        self.subject_ids = subject_ids

    # Build file list for DINO features
        if subject_ids is not None:
            self.feature_files = []
            for sid in subject_ids:
                pattern = os.path.join(root_dir, sid, "DINO_Features", plane, "slice*.pt")
                subj_files = sorted(glob.glob(pattern))
                if len(subj_files) == 0:
                    print(f"Warning: No DINO feature files found for subject {sid} ({pattern})")
                self.feature_files.extend(subj_files)
        else:
            self.feature_files = sorted(glob.glob(os.path.join(root_dir, "s*", "DINO_Features", plane, "slice*.pt")))

        if len(self.feature_files) == 0:
            raise ValueError(f"No DINO feature files found in {root_dir} for plane '{plane}'")

        # Derive skip paths by parallel directory structure
        self.skip_files = [
            f.replace("DINO_Features", "DINOSKIP_Features")
            for f in self.feature_files
        ]

        # Check existence
        missing_skips = [sf for sf in self.skip_files if not os.path.exists(sf)]
        if len(missing_skips) > 0:
            print(f"Warning: {len(missing_skips)} slices missing skip features.")

        unique_subjects = {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f)))) for f in self.feature_files}
        print(f"FilteredDINOFeatureAndSkipDataset: {len(self.feature_files)} slices from subjects: {unique_subjects}")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        # Load main DINO feature
        feat_path = self.feature_files[idx]
        feat_data = torch.load(feat_path, map_location="cpu")

        features = feat_data["features"].float()   # (num_patches, feature_dim)
        mask = feat_data["mask"].long()            # (H, W)
        meta = feat_data["meta"]

        # Load skip features (if present)
        skip_path = self.skip_files[idx]
        skip_data = torch.load(skip_path, map_location="cpu") if os.path.exists(skip_path) else {"skip_features": []}
        skip_features = [sf.float() for sf in skip_data.get("skip_features", [])]

        return {
            "features": features,
            "skip_features": skip_features,
            "mask": mask,
            "meta": meta,
        }
    


class FilteredDINOFeatureAndSkipDataset2(Dataset):
    """
    Slice-level dataset that loads both DINO features and DINO skip features.
    Replaces layer 3 and 6 skip features with layer 2 and 5 from DINOSKIP_Features_additional.
    """

    def __init__(self, root_dir: str, plane: str = "axial", subject_ids: list[str] = None):
        self.root_dir = root_dir
        self.plane = plane
        self.subject_ids = subject_ids

        # Build list of DINO feature files
        if subject_ids is not None:
            self.feature_files = []
            for sid in subject_ids:
                pattern = os.path.join(root_dir, sid, "DINO_Features", plane, "slice*.pt")
                subj_files = sorted(glob.glob(pattern))
                if len(subj_files) == 0:
                    print(f"⚠️ Warning: No DINO feature files found for subject {sid} ({pattern})")
                self.feature_files.extend(subj_files)
        else:
            self.feature_files = sorted(glob.glob(os.path.join(root_dir, "s*", "DINO_Features", plane, "slice*.pt")))

        if len(self.feature_files) == 0:
            raise ValueError(f"No DINO feature files found in {root_dir} for plane '{plane}'")

        # Derive skip feature paths
        self.skip_files_main = [
            f.replace("DINO_Features", "DINOSKIP_Features") for f in self.feature_files
        ]
        self.skip_files_additional = [
            f.replace("DINO_Features", "DINOSKIP_Features_additional") for f in self.feature_files
        ]

        # Check existence of main skip features
        missing_skips = [sf for sf in self.skip_files_main if not os.path.exists(sf)]
        if len(missing_skips) > 0:
            print(f" Warning: {len(missing_skips)} slices missing main skip features.")

        unique_subjects = {
            os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
            for f in self.feature_files
        }
        print(f"FilteredDINOFeatureAndSkipDataset2: {len(self.feature_files)} slices from subjects: {unique_subjects}")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        #  Load main DINO features 
        feat_path = self.feature_files[idx]
        feat_data = torch.load(feat_path, map_location="cpu")

        features = feat_data["features"].float()
        mask = feat_data["mask"].long()
        meta = feat_data["meta"]

        # Load main skip features 
        skip_path_main = self.skip_files_main[idx]
        skip_data_main = torch.load(skip_path_main, map_location="cpu") if os.path.exists(skip_path_main) else {}
        skip_features_main = skip_data_main.get("skip_features", [])

        # Load additional skip features (for replacement
        skip_path_add = self.skip_files_additional[idx]
        skip_data_add = torch.load(skip_path_add, map_location="cpu") if os.path.exists(skip_path_add) else {}
        skip_features_add = skip_data_add.get("skip_features", [])

        # Merge skip features 
        # Expected: original had [layer3, layer6, layer9]
        # Replacement: use layer2 (idx 0) for 3, layer5 (idx 1) for 6, keep 9
        if skip_features_main:
            merged_skips = []
            if len(skip_features_add) >= 2:
                # Replace layers 3 and 6
                merged_skips.append(skip_features_add[0].float())  # replaces layer 3
                merged_skips.append(skip_features_add[1].float())  # replaces layer 6
            elif len(skip_features_add) == 1:
                merged_skips.append(skip_features_add[0].float())
                merged_skips.append(skip_features_main[1].float()) if len(skip_features_main) > 1 else None
            else:
                merged_skips.extend(skip_features_main[:2])
            
            # Keep layer 9 (last one from main)
            if len(skip_features_main) >= 3:
                merged_skips.append(skip_features_main[2].float())
        else:
            merged_skips = [sf.float() for sf in skip_features_add]

        return {
            "features": features,
            "skip_features": merged_skips,
            "mask": mask,
            "meta": meta,
        }



class Slicedataset(Dataset):
    def __init__(self, data_dir, plane, subjects):
        self.data_dir = data_dir
        self.plane = plane
        self.subjects = subjects
        
        self.samples = []
        for subject in subjects:
            subj_dir = os.path.join(self.data_dir, subject, "ImgSlices", self.plane)
            if not os.path.exists(subj_dir):
                print(f"Warning: Directory {subj_dir} does not exist. Skipping subject {subject}.")
                continue
            # get how many .pt files are in subj_dir
            for name in sorted(os.listdir(subj_dir)):
                if name.endswith(".pt"):
                    id = int(name.replace("slice", "").replace(".pt", ""))
                    self.samples.append({"subject": subject, "slice_file": os.path.join(subj_dir, name), "idx": id})


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        slice_tensor = torch.load(sample_info["slice_file"])
        id = int(os.path.basename(sample_info["slice_file"]).replace("slice", "").replace(".pt", ""))
        image = slice_tensor["image"].float()
        mask = slice_tensor["mask"].long()

        return {"image": image, "mask": mask , "subject": sample_info["subject"],"idx": id, "plane": self.plane}


class SlicedatasetArgumentation(Dataset):
    def __init__(self, data_dir, plane, subjects, transform=None):
        self.data_dir = data_dir
        self.plane = plane
        self.subjects = subjects
        self.transform = transform

        self.samples = []
        for subject in subjects:
            subj_dir = os.path.join(self.data_dir, subject, "ImgSlices", self.plane)
            if not os.path.exists(subj_dir):
                print(f"Warning: Directory {subj_dir} does not exist. Skipping subject {subject}.")
                continue
            # get how many .pt files are in subj_dir
            for name in sorted(os.listdir(subj_dir)):
                if name.endswith(".pt"):
                    id = int(name.replace("slice", "").replace(".pt", ""))
                    self.samples.append({"subject": subject, "slice_file": os.path.join(subj_dir, name), "idx": id})


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        slice_tensor = torch.load(sample_info["slice_file"])
        id = int(os.path.basename(sample_info["slice_file"]).replace("slice", "").replace(".pt", ""))
        image = slice_tensor["image"].float().squeeze(0)  # remove batch dimension
        mask = slice_tensor["mask"].long().squeeze(0)    # remove batch dimension

        image = tv_tensors.Image(image)
        mask  = tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        image = image.unsqueeze(0)  # add batch dimension back
        mask = mask.unsqueeze(0)   # add batch dimension back

    

        return {"image": image, "mask": mask , "subject": sample_info["subject"],"idx": id, "plane": self.plane}

class CrossSlicedataset(Dataset):
    def __init__(self, data_dir, plane, subjects, K=3):
        """
        data_dir: root directory
        plane: axial / sagittal / coronal
        subjects: list of subject IDs
        K: number of slices to pack together (must be >= 1)
        """
        assert K >= 1, "K must be >= 1"
        assert K % 2 == 1, "K must be odd (center slice must exist)"

        self.data_dir = data_dir
        self.plane = plane
        self.subjects = subjects
        self.K = K
        self.radius = K // 2  # number of slices on each side

        # build subject -> slice list and global index
        self.subject_to_slices = {}
        self.index = []  # list of (subject, slice_idx)

        for subject in subjects:
            subj_dir = os.path.join(data_dir, subject, "ImgSlices", plane)
            if not os.path.isdir(subj_dir):
                print(f"Warning: Missing directory {subj_dir}, skipping {subject}")
                continue

            slice_files = sorted([f for f in os.listdir(subj_dir) if f.endswith(".pt")])
            full_paths = [os.path.join(subj_dir, f) for f in slice_files]

            self.subject_to_slices[subject] = full_paths

            for i in range(len(full_paths)):
                self.index.append((subject, i))

    def __len__(self):
        return len(self.index)

    def load_slice(self, subject, slice_idx):
        """Loads (image, mask) from a .pt file."""
        path = self.subject_to_slices[subject][slice_idx]
        data = torch.load(path)
        return data["image"].float(), data["mask"].long()

    def __getitem__(self, idx):
        subject, center_idx = self.index[idx]
        slice_list = self.subject_to_slices[subject]
        N = len(slice_list)

        imgs = []
        radius = self.radius

        # build index window [center-radius, center+radius]
        for offset in range(-radius, radius + 1):
            si = center_idx + offset

            # handle boundaries: replicate nearest valid slice
            if si < 0:
                si = 0
            elif si >= N:
                si = N - 1

            img, _ = self.load_slice(subject, si)
            imgs.append(img)  # each img is [1,H,W]

        # stack into channels → [K, H, W]
        image_stack = torch.cat(imgs, dim=0)

        # load center slice mask
        _, mask = self.load_slice(subject, center_idx)

        return {
            "image": image_stack,   # [K, H, W]
            "mask": mask,           # [1, H, W]
            "subject": subject,
            "slice_idx": center_idx,
            "plane": self.plane,
            "slice_indexes": list(range(center_idx - radius, center_idx + radius + 1)),
        }

class VolSlicedataset(Dataset):
    def __init__(self, data_dir, plane, subjects):
        self.data_dir = data_dir
        self.plane = plane
        self.subjects = subjects

        self.samples = []
        for subject in subjects:
            subj_dir = os.path.join(self.data_dir, subject, "ImgSlices", self.plane)
            if not os.path.exists(subj_dir):
                print(f"Warning: Directory {subj_dir} does not exist. Skipping subject {subject}.")
                continue

            slice_files = sorted([
                os.path.join(subj_dir, name)
                for name in os.listdir(subj_dir)
                if name.endswith(".pt")
            ])

            if slice_files:
                self.samples.append({"subject": subject, "slice_files": slice_files})

            # print number of subject found
        print(f"VolSlicedataset: Found {len(self.samples)} subjects in {data_dir} for plane {plane}.")

    def __len__(self):
        # Now, one item = one subject
        return len(self.samples)

    

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        subject = sample_info["subject"]
        return self._load_subject_slices(subject, sample_info["slice_files"])

    def _load_subject_slices(self, subject, slice_files):
        """Internal helper to load all slices for a given subject."""
        images, masks = [], []
        for sf in slice_files:
            slice_tensor = torch.load(sf)
            images.append(slice_tensor["image"].float())
            masks.append(slice_tensor["mask"].long())

        images = torch.stack(images, dim=0)  # (num_slices, C, H, W)
        masks = torch.stack(masks, dim=0)    # (num_slices, H, W)
        return {"images": images, "masks": masks, "subject": subject}

    def get_subject(self, subject_id):
        """Return all slice tensors for a specific subject ID."""
        for sample in self.samples:
            if sample["subject"] == subject_id:
                return self._load_subject_slices(sample["subject"], sample["slice_files"])
        raise ValueError(f"Subject '{subject_id}' not found in dataset.")


class AnyUpFeaturesdataset(Dataset):
    def __init__(self, data_dir, subjects):

        self.data_dir = data_dir
        self.plane = plane
        self.subjects = subjects

        self.samples = []
        for subject in self.subjects:
            upsampled_dir = os.path.join(
                self.data_dir,
                subject,
                "upsampled"
            )
            # get all sub folders in the upsampled_dir
            planes_found = [
                name for name in os.listdir(upsampled_dir)

                if os.path.isdir(os.path.join(upsampled_dir, name))
            ]
            for plane in planes_found:
                plane_dir = os.path.join(
                    upsampled_dir,
                    plane
                )
                files = sorted(
                    glob.glob(os.path.join(plane_dir, "upfeat_slice*.pt")),
                    key=self._extract_slice_idx
                )
                
                for f in files:
                    slice_idx = self._extract_slice_idx(f)  
                    img_file  = os.path.join(self.data_dir, subject, "ImgSlices", self.plane,"slice{:03d}.pt".format(slice_idx))
                    self.samples.append({
                       "subject": subject,
                       "img_file": img_file,
                       "upsampled_features": f,
                       "slice_idx": slice_idx
                    })
    
    def _extract_slice_idx(self, filepath):
        match = re.search(r"upfeat_slice(\d+)\.pt", filepath)
        if match:
            return int(match.group(1))
        else:
            return -1  # or raise an error if preferred
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        slice_tensor = torch.load(sample_info["img_file"])
        image = slice_tensor["image"].float()
        mask = slice_tensor["mask"].long()

        upfeat_tensor = torch.load(sample_info["upsampled_features"])
        upsampled_features = upfeat_tensor["upsampled_features"].float()

        return {
            "image": image,                            # (C, H, W)
            "mask": mask,                              # (H, W)
            "subject": sample_info["subject"],
            "slice_idx": sample_info["slice_idx"],
            "upsampled_features": upsampled_features   # (feature_dim, H', W')
        }
    
class CuratedSliceDataset(torch.utils.data.Dataset):
    def __init__(self, slice_data_dir, plane):
        self.slice_files = glob.glob(os.path.join(slice_data_dir, plane, "*.pt"))
    
    def __len__(self):
        return len(self.slice_files)
    
    def __getitem__(self, idx):
        slice_data = torch.load(self.slice_files[idx])
        return slice_data
    
class CuratedSliceDatasetArgumented(torch.utils.data.Dataset):
    def __init__(self, slice_data_dir, plane, transform=None):
        self.slice_files = glob.glob(os.path.join(slice_data_dir, plane, "*.pt"))
        self.transform = transform

    def __len__(self):
        return len(self.slice_files)
    
    def __getitem__(self, idx):
        slice_data = torch.load(self.slice_files[idx])
        image = slice_data["image"].float().squeeze(0)  # remove batch dimension
        mask = slice_data["mask"].long().squeeze(0)    # remove batch dimension

        image = tv_tensors.Image(image)
        mask  = tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        
        
        slice_data["image"] = image.unsqueeze(0)  # add batch dimension back
        slice_data["mask"]  = mask.unsqueeze(0)   # add batch dimension back

        return slice_data