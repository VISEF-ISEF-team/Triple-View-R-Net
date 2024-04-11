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


def one_hot_encode(mask):
    unique_value = [0.0,
                    500.0,
                    600.0,
                    420.0,
                    550.0,
                    205.0,
                    820.0,
                    850.0]

    output = [np.where(mask == x, 1, 0) for x in unique_value]
    output = np.stack(output, axis=0)
    return output


def normalize_image_intensity_range(img):
    """Normalise for VHSCDD"""
    # HOUNSFIELD_MAX = 4000
    # HOUNSFIELD_MIN = 0

    HOUNSFIELD_MAX = np.max(img)
    HOUNSFIELD_MIN = np.min(img)

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


class MMWHS3D(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(self.images_path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        image = nib.load(self.images_path[index]).get_fdata()
        mask = nib.load(self.masks_path[index]).get_fdata()

        image = normalize_image_intensity_range(image)
        reshaped_image = resize(
            image, (128, 128, 128), anti_aliasing=True, preserve_range=True, order=1)
        reshaped_image = np.expand_dims(
            reshaped_image, axis=0).astype(np.float32)

        # convert unique value in mask to class
        mask = one_hot_encode(mask)
        reshaped_mask = resize(mask, (8, 128, 128, 128),
                               anti_aliasing=True, preserve_range=True, order=1)
        reshaped_mask = np.argmax(reshaped_mask, axis=0)
        reshaped_mask = np.expand_dims(reshaped_mask, 0)

        return reshaped_image, reshaped_mask


def load_dataset(image_path, mask_path, split=0.1):
    images = sorted(glob(os.path.join(image_path, "*_image.nii.gz")))
    masks = sorted(glob(os.path.join(mask_path, "*_label.nii.gz")))

    split_size = int(split * len(images))

    x_train, x_val = train_test_split(
        images, test_size=split_size, random_state=42)

    y_train, y_val = train_test_split(
        masks, test_size=split_size, random_state=42)

    return (x_train, y_train), (x_val, y_val)


def get_loaders():
    image_path = "../data_for_training/MMWHS/ct_train/"
    mask_path = "../data_for_training/MMWHS/ct_train/"

    (x_train, y_train), (x_val, y_val) = load_dataset(image_path, mask_path)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = MMWHS3D(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=2)

    val_dataset = MMWHS3D(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=2)

    return (train_loader, val_loader)


def main():
    train_loader, val_loader = get_loaders()

    for x, y in train_loader:
        print(f"Image max: {x.max()} - min: {x.min()}", end=" ")
        print(f"Unique label in mask: {np.unique(y.detach().numpy())}")


if __name__ == "__main__":
    main()
