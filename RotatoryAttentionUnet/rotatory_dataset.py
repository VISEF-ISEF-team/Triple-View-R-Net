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
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms


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

    def convert_label_to_class(self, mask):
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

    def __getitem__(self, index):
        image = nib.load(self.images_path[index]).get_fdata()
        mask = nib.load(self.masks_path[index]).get_fdata()

        image = self.normalize_image_intensity_range(image)
        self.convert_label_to_class(mask)

        return image, mask


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
    root = "../data_for_training/MMWHS/"

    root_images = os.path.join(root, "ct_train", "images")
    root_labels = os.path.join(root, "ct_train", "masks")

    (x_train, y_train), (x_val, y_val) = load_dataset(root_images, root_labels)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = RotCAttTransDense_Dataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=None, shuffle=True, num_workers=6)

    val_dataset = RotCAttTransDense_Dataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=None,
                            shuffle=False, num_workers=6)

    return (train_loader, val_loader)


def get_slice_from_volumetric_data(image_volume, mask_volume, start_idx, num_slice=12, train_transform=None, val_transform=None):

    end_idx = start_idx + num_slice

    images = torch.empty(num_slice, 1, 256, 256)
    masks = torch.empty(num_slice, 1, 256, 256)

    for i in range(start_idx, end_idx, 1):
        image = image_volume[:, :, i].numpy()
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.uint8)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)

        if train_transform != None:
            image = train_transform(image)

        elif val_transform != None:
            image = val_transform(image)

        images[i - start_idx, :, :, :] = image

        mask = mask_volume[:, :, i].long()
        mask = F.one_hot(mask, num_classes=8)
        mask = mask.numpy()
        mask = resize(mask, (256, 256, 8),
                      preserve_range=True, anti_aliasing=True)
        mask = torch.from_numpy(mask)
        mask = torch.argmax(mask, dim=-1)
        mask = torch.unsqueeze(mask, dim=0)

        masks[i - start_idx, :, :, :] = mask

    return images, masks


def duplicate_open_end(x):
    first_slice = x[:, :, 0].unsqueeze(2)
    last_slice = x[:, :, -1].unsqueeze(2)
    x = torch.cat((first_slice, x, last_slice), dim=2)

    return x


def duplicate_end(x):
    last_slice = x[:, :, -1].unsqueeze(2)
    x = torch.cat((x, last_slice), dim=2)

    return x


def main():
    train_loader, val_loader = get_loaders()

    train_transform_trivial = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins=5),
    ])

    for x, y in train_loader:

        x = duplicate_open_end(x)
        y = duplicate_open_end(y)

        length = x.shape[-1]

        print(f"Image Volume: {x.shape} || Mask Volume: {y.shape}")

        for i in range(0, length, 7):

            if i + 8 >= length:
                num_slice = length - i
                if num_slice < 3:
                    for i in range(3 - num_slice):
                        x = duplicate_end(x)
                        y = duplicate_end(y)

                    num_slice = 3

            else:
                num_slice = 8

            images, masks = get_slice_from_volumetric_data(
                x, y, i, num_slice, train_transform=train_transform_trivial)

            print(images.shape, masks.shape)

        print("-" * 30)


if __name__ == "__main__":
    main()