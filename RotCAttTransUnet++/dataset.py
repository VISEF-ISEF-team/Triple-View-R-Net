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
    root = "../data_for_training/Synapse/RawData/Training"

    root_images = os.path.join(root, "img")
    root_labels = os.path.join(root, "label")

    (x_train, y_train), (x_val, y_val) = load_dataset(root_images, root_labels)

    print(f"Train: {len(x_train)} || {len(y_train)}")
    print(f"Val: {len(x_val)} || {len(y_val)}")

    train_dataset = RotCAttTransDense_Dataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=None, shuffle=True, num_workers=4)

    val_dataset = RotCAttTransDense_Dataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=None,
                            shuffle=False, num_workers=4)

    return (train_loader, val_loader)


def get_slice_from_volumetric_data(image_volume, mask_volume, start_idx, num_slice=8):

    end_idx = start_idx + num_slice

    images = torch.empty(num_slice, 1, 256, 256)
    masks = torch.empty(num_slice, 1, 256, 256)

    for i in range(start_idx, end_idx, 1):
        image = image_volume[:, :, i].numpy()
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)

        images[i - start_idx, :, :, :] = image

        mask = mask_volume[:, :, i].long()
        mask = F.one_hot(mask, num_classes=14)
        mask = mask.numpy()
        mask = resize(mask, (256, 256, 14),
                      preserve_range=True, anti_aliasing=True)
        mask = torch.from_numpy(mask)
        mask = torch.argmax(mask, dim=-1)
        mask = torch.unsqueeze(mask, dim=0)

        masks[i - start_idx, :, :, :] = mask

    return images, masks


def main():
    train_loader, val_loader = get_loaders()

    for x, y in train_loader:
        first_slice = x[:, :, 0].unsqueeze(2)
        last_slice = x[:, :, -1].unsqueeze(2)
        x = torch.cat((first_slice, x, last_slice), dim=2)

        first_slice = y[:, :, 0].unsqueeze(2)
        last_slice = y[:, :, -1].unsqueeze(2)
        y = torch.cat((first_slice, y, last_slice), dim=2)

        length = x.shape[-1]

        print(f"Image Volume: {x.shape} || Mask Volume: {y.shape}")

        for i in range(0, length, 8):

            if i + 8 >= length:
                num_slice = length - i
            else:
                num_slice = 8

            images, masks = get_slice_from_volumetric_data(x, y, i, num_slice)

            print(images.shape, masks.shape)

        break


if __name__ == "__main__":
    main()
