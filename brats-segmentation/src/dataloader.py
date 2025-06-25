
import os 
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import albumentations as A

class BraTSDataset(Dataset):
    def __init__(self, patients, transform=None, mode='train', slice_selection_method='all'):
        self.patients = patients
        self.transform = transform
        self.mode = mode
        self.slice_selection_method = slice_selection_method
        
        self.slices = []
        
        for patient in self.patients:
            patient_id = patient['id']
            patient_path = patient['path']
            grade = patient['grade']
            
            t1_path = os.path.join(patient_path, f"{patient_id}_t1.nii")
            t1ce_path = os.path.join(patient_path, f"{patient_id}_t1ce.nii")
            t2_path = os.path.join(patient_path, f"{patient_id}_t2.nii")
            flair_path = os.path.join(patient_path, f"{patient_id}_flair.nii")
            seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")

            if not all(os.path.exists(p) for p in [t1_path, t1ce_path, t2_path, flair_path, seg_path]):
                t1_path = os.path.join(patient_path, f"{patient_id}_T1.nii.gz")
                t1ce_path = os.path.join(patient_path, f"{patient_id}_T1CE.nii.gz")
                t2_path = os.path.join(patient_path, f"{patient_id}_T2.nii.gz")
                flair_path = os.path.join(patient_path, f"{patient_id}_FLAIR.nii.gz")
                seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
            if not all(os.path.exists(p) for p in [t1_path, t1ce_path, t2_path, flair_path, seg_path]):
                print(f"Skipping {patient_id}: Missing required files")
                continue
                
            seg_img = nib.load(seg_path).get_fdata()
            
            if slice_selection_method == 'all':
                for slice_idx in range(seg_img.shape[2]):
                    if slice_idx >= seg_img.shape[2] * 0.1 and slice_idx <= seg_img.shape[2] * 0.9:
                        self.slices.append({
                            'patient': patient_id,
                            'grade': grade,
                            'slice_idx': slice_idx,
                            't1_path': t1_path,
                            't1ce_path': t1ce_path,
                            't2_path': t2_path,
                            'flair_path': flair_path,
                            'seg_path': seg_path
                        })
            elif slice_selection_method == 'tumor_only':
                for slice_idx in range(seg_img.shape[2]):
                    if np.any(seg_img[:, :, slice_idx] > 0):
                        self.slices.append({
                            'patient': patient_id,
                            'grade': grade,
                            'slice_idx': slice_idx,
                            't1_path': t1_path,
                            't1ce_path': t1ce_path,
                            't2_path': t2_path,
                            'flair_path': flair_path,
                            'seg_path': seg_path
                        })
            else:
                raise ValueError(f"Invalid slice selection method: {slice_selection_method}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_data = self.slices[idx]
        patient = slice_data['patient']
        slice_idx = slice_data['slice_idx']
        
        t1 = nib.load(slice_data['t1_path']).get_fdata()[:, :, slice_idx]
        t1ce = nib.load(slice_data['t1ce_path']).get_fdata()[:, :, slice_idx]
        t2 = nib.load(slice_data['t2_path']).get_fdata()[:, :, slice_idx]
        flair = nib.load(slice_data['flair_path']).get_fdata()[:, :, slice_idx]
        
        seg = nib.load(slice_data['seg_path']).get_fdata()[:, :, slice_idx]
        
        t1 = self._preprocess(t1)
        t1ce = self._preprocess(t1ce)
        t2 = self._preprocess(t2)
        flair = self._preprocess(flair)
        
        image = np.stack([t1, t1ce, t2, flair], axis=0).astype(np.float32)
        
        mask = np.zeros((3, *seg.shape), dtype=np.float32)
        mask[0, seg == 1] = 1  # NCR/NET
        mask[1, seg == 2] = 1  # ED
        mask[2, seg == 4] = 1  # ET
        
        if self.transform:
            transformed = self.transform(image=image.transpose(1, 2, 0), mask=mask.transpose(1, 2, 0))
            image = transformed['image'].transpose(2, 0, 1)
            mask = transformed['mask'].transpose(2, 0, 1)
        
        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask),
            'patient': patient,
            'slice': slice_idx
        }
    
    def _preprocess(self, img):
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / std
        
        img = np.clip(img, -5, 5)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        return img

def get_train_transforms(size=240):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
    ])

def get_val_transforms(size=240):
    return A.Compose([
        A.Resize(size, size),
    ])
