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
from fct_utils import OneHotResizing


class DeepSupervisionDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(self.images_path)
        self.resizer = OneHotResizing(
            num_class=8, categories=[0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float16)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.expand_dims(
            image, axis=0).astype(np.float32)

        mask = np.load(self.masks_path[index])
        mask1 = self.resizer.one_hot_resize(mask, (8, 128, 128))
        mask2 = self.resizer.one_hot_resize(mask, (8, 64, 64))

        mask = np.expand_dims(mask, axis=0)
        mask1 = np.expand_dims(mask1, axis=0)
        mask2 = np.expand_dims(mask2, axis=0)

        mask_list = [mask2, mask1, mask]

        return image, mask_list


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
    image_path = "../UNETR_MMWHS/files/images/"
    mask_path = "../UNETR_MMWHS/files/masks/"

    (x_train, y_train), (x_val, y_val) = load_dataset(image_path, mask_path)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = DeepSupervisionDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4)

    x, y = train_dataset[0]

    val_dataset = DeepSupervisionDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=False, num_workers=4)

    return (train_loader, val_loader)


def main():
    train_loader, val_loader = get_loaders()

    for x, y in train_loader:
        for i in y:
            print(i.shape)


if __name__ == "__main__":

    main()
