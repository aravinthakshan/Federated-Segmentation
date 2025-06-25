from medpy.metric import binary
import nibabel as nib
import matplotlib.pyplot as plt

### Metric Calcs

def dice_coefficient(y_true, y_pred):
    return binary.dc(y_pred, y_true)

def hausdorff_distance_95(y_true, y_pred):
    return binary.hd95(y_pred, y_true)


### Visualizations

def visualize_nifti(file_path):
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    slice_index = data.shape[2] // 2 # Middle slice
    plt.figure(figsize=(8, 8))
    plt.imshow(data[:, :, slice_index], cmap='gray')
    plt.title(f"Visualization of {file_path.split('/')[-1]} - Slice {slice_index}")
    plt.colorbar()
    plt.axis('off')
    plt.show()

file_paths = [
    # /kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training
    "/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_10_1/BraTS19_2013_10_1_flair.nii",
    "/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_10_1/BraTS19_2013_10_1_t1.nii",
    "/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_10_1/BraTS19_2013_10_1_seg.nii",
    "/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_10_1/BraTS19_2013_10_1_t1ce.nii",
    "/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_10_1/BraTS19_2013_10_1_t2.nii"
]

# Visualize each file
for file_path in file_paths:
    visualize_nifti(file_path)
