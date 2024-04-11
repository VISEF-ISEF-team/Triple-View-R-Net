import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt


def normalize_image_intensity_range(img):
    HOUNSFIELD_MAX = 4000
    HOUNSFIELD_MIN = 0
    HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX

    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE


def convert_label_to_class(mask):
    lookup_table = {
        0.0: 0.0,
        500.0: 1.0,
        600.0: 2.0,
        420.0: 3.0,
        550.0: 4.0,
        205.0: 5.0,
        820.0: 6.0,
        850.0: 7.0,
    }

    for i in np.unique(mask):
        mask[mask == i] = lookup_table[i]


class SynapseDataset3D(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        image = nib.load(self.images_path[index]).get_fdata()
        mask = nib.load(self.masks_path[index]).get_fdata()

        reshaped_image = resize(image, (256, 256, 16),   anti_aliasing=False)
        reshaped_image = reshaped_image / 255.0
        reshaped_image = np.expand_dims(reshaped_image, axis=0)
        reshaped_image = np.round(reshaped_image, 2)
        reshaped_image = reshaped_image.astype(np.float32)

        reshaped_mask = resize(mask, (256, 256, 16), anti_aliasing=True)
        reshaped_mask = np.expand_dims(reshaped_mask, axis=0)
        reshaped_mask = reshaped_mask.astype(np.uint8)

        return reshaped_image, reshaped_mask


def load_dataset(image_path, mask_path, split=0.2):
    images = sorted(glob(os.path.join(image_path, "*_image.nii.gz")))
    masks = sorted(glob(os.path.join(mask_path, "*_label.nii.gz")))

    split_size = int(split * len(images))

    x_train, x_test = train_test_split(
        images, test_size=split_size, random_state=42)

    y_train, y_test = train_test_split(
        masks, test_size=split_size, random_state=42)

    x_train, x_val = train_test_split(
        x_train, test_size=split_size, random_state=42)
    y_train, y_val = train_test_split(
        y_train, test_size=split_size, random_state=42)

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def resizing_check():
    # process image
    image = nib.load(
        "../").get_fdata()
    reshaped_image = resize(image, (256, 256, 16), anti_aliasing=True)
    reshaped_image = reshaped_image / 255.0
    reshaped_image = np.round(reshaped_image, 2)
    reshaped_image = reshaped_image.astype(np.float32)

    print(reshaped_image.min(), reshaped_image.max(),
          len(np.unique(reshaped_image)))

    # process mask
    mask = nib.load(
        "../data_for_training/3d_training_masks/DET0000101_avg_seg.nii.gz").get_fdata()
    reshaped_mask = resize(mask, (256, 256, 16), anti_aliasing=True)
    reshaped_mask = reshaped_mask.astype(np.uint8)
    print(reshaped_mask.min(), reshaped_mask.max(),
          len(np.unique(reshaped_mask)))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
    axes[0].imshow(reshaped_image[:, :, 6])
    axes[0].set_xlabel("image")
    axes[1].imshow(reshaped_mask[:, :, 6])
    axes[1].set_xlabel("mask")

    plt.show()


def get_loaders():
    image_path = os.path.join(
        "..", "data_for_training", "MMWHS", "ct_train")
    mask_path = os.path.join("..", "data_for_training",
                             "MMWHS", "ct_train")

    (x_train, y_train), (x_test, y_test), (x_val,
                                           y_val) = load_dataset(image_path, mask_path)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Test: {len(x_test)} || {len(y_test)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = SynapseDataset3D(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=2, num_workers=2, shuffle=True)

    val_dataset = SynapseDataset3D(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=2,
                            num_workers=2, shuffle=False)

    test_dataset = SynapseDataset3D(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2,
                             num_workers=2, shuffle=False)

    return (train_loader, val_loader, test_loader)


def main():
    train_loader, val_loader, test_loader = get_loaders()

    for x, y in train_loader:
        x = x.numpy()
        print(x.min(), x.max(), len(np.unique(x)))


if __name__ == "__main__":
    # main()
    # resizing_check()
    get_loaders()
