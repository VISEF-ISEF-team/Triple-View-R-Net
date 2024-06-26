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
from patchify import patchify


def normalize_image_intensity_range(img):
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


class MMWHS2D(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(self.images_path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.load(self.masks_path[index])
        patch_shape = (16, 16)

        # process image
        image = image / 255.0
        image = image.astype(np.float32)
        patches = patchify(image, patch_size=patch_shape, step=16)
        patches = np.reshape(patches, (16*16, 16*16))

        # process mask
        mask = np.expand_dims(mask, 0).astype(np.uint8)

        return patches, mask


def load_dataset(image_path, mask_path, split=0.2):
    images = sorted(glob(os.path.join(image_path, "*.png")))
    masks = sorted(glob(os.path.join(mask_path, "*.npy")))

    split_size = int(split * len(images))

    x_train, x_val = train_test_split(
        images, test_size=split_size, random_state=42)

    y_train, y_val = train_test_split(
        masks, test_size=split_size, random_state=42)

    return (x_train, y_train), (x_val, y_val)


def get_loaders():
    image_path = "./files/images/"
    mask_path = "./files/masks/"

    (x_train, y_train), (x_val, y_val) = load_dataset(image_path, mask_path)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = MMWHS2D(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4)

    val_dataset = MMWHS2D(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=False, num_workers=4)

    return (train_loader, val_loader)


def main():
    train_loader, val_loader = get_loaders()

    for x, y in train_loader:
        print(f"Image max: {x.max()} - min: {x.min()}", end=" ")
        print(
            f"Unique label in mask: {np.unique(y.detach().numpy())}", end=" ")
        print(f"Input shape: {x.shape} || Mask shape: {y.shape}")


if __name__ == "__main__":
    # image = cv2.imread(
    #     "./files/images/heart0-slice000_axial.png", cv2.IMREAD_GRAYSCALE)
    # patch_shape = (16, 16)

    # # process image
    # image = image / 255.0
    # image = image.astype(np.float32)
    # patches = patchify(image, patch_size=patch_shape, step=16)

    # print(patches.shape)

    main()
