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


class RotCAttTransDense_Dataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(self.images_path)

    def __len__(self):
        return self.n_samples

    def normalize_image_intensity_range(self, img):
        HOUNSFIELD_MAX = np.max(img)
        HOUNSFIELD_MIN = np.min(img)
        HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

        img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
        img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX

        return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

    def __getitem__(self, index):
        image = nib.load(self.images_path[index]).get_fdata()
        mask = nib.load(self.masks_path[index]).get_fdata().astype(np.uint8)

        image = self.normalize_image_intensity_range(image)

        return (self.images_path[index], self.masks_path[index], image, mask)


def load_dataset(image_path, mask_path, split=0.2):
    images = sorted(glob(os.path.join(image_path, "*.nii.gz")))
    masks = sorted(glob(os.path.join(mask_path, "*.nii.gz")))

    split_size = int(split * len(images))

    x_train, x_val = train_test_split(
        images, test_size=split_size, random_state=42)

    y_train, y_val = train_test_split(
        masks, test_size=split_size, random_state=42)

    return (x_train, y_train), (x_val, y_val)


def get_loaders():
    root = "../data_for_training/Synapse/RawData/Training"

    root_images = os.path.join(root, "img")
    root_labels = os.path.join(root, "label")

    (x_train, y_train), (x_val, y_val) = load_dataset(root_images, root_labels)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = RotCAttTransDense_Dataset(x_train, y_train)
    # train_loader = DataLoader(
    #     train_dataset, batch_size=4, shuffle=True, num_workers=4)

    val_dataset = RotCAttTransDense_Dataset(x_val, y_val)
    # val_loader = DataLoader(val_dataset, batch_size=4,
    #                         shuffle=False, num_workers=4)

    # return (train_loader, val_loader)

    im, label, x, y = train_dataset[0]

    print(f"Path: {im} || X: {x.shape}")
    print(F"Path: {label} || Y: {y.shape}")


def main():
    train_loader, val_loader = get_loaders()

    for x, y in train_loader:
        print(f"{x.shape} || {y.shape}")


if __name__ == "__main__":
    get_loaders()
