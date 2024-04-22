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
import torch.nn as nn
import torch.nn.functional as F


class CustomDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.softmax(inputs, dim=1)

        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class UNET_ATTENTION_ORIGINAL_DATASET(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(self.images_path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.expand_dims(
            image, axis=0).astype(np.float32)

        mask = np.load(self.masks_path[index])
        mask = torch.from_numpy(mask)
        mask = F.one_hot(mask, num_classes=8)
        mask = mask.numpy()
        mask = resize(mask, (128, 128, 8),
                      preserve_range=True, anti_aliasing=True)
        mask = torch.from_numpy(mask)
        mask = torch.argmax(mask, dim=-1)
        mask = torch.unsqueeze(mask, dim=0)

        return image, mask


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
    image_path = "../data_for_training/MMWHS/images/"
    mask_path = "../data_for_training/MMWHS/masks/"

    (x_train, y_train), (x_val, y_val) = load_dataset(image_path, mask_path)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = UNET_ATTENTION_ORIGINAL_DATASET(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=20, shuffle=True, num_workers=4)

    val_dataset = UNET_ATTENTION_ORIGINAL_DATASET(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=20,
                            shuffle=False, num_workers=4)

    return (train_loader, val_loader)


def main():
    train_loader, val_loader = get_loaders()

    for x, y in train_loader:
        print(f"{x.shape} || {y.shape}")


if __name__ == "__main__":
    # a = nib.load(
    #     "../data_for_training/MMWHS/ct_train/ct_train_1001_label.nii.gz").get_fdata()
    # print("Before converting")
    # print(np.unique(a))
    # print("-" * 100)
    # print("After converting")
    # convert_label_to_class(a)
    # print(np.unique(a))
    main()
