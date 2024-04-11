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


class ROTATORY_MMWHS(Dataset):
    def __init__(self, root_images, root_masks, num_unique: list):
        """
        num_unique: list of unique heart index for current dataset
        """
        self.images = []
        self.masks = []

        for n in num_unique:
            image_paths = sorted(
                glob(os.path.join(root_images, f"heart{n}-*_axial.png")))
            mask_paths = sorted(
                glob(os.path.join(root_masks, f"heartmaskencode{n}-*_axial.npy")))

            assert len(image_paths) == len(
                mask_paths), f"Length image: {len(image_paths)} is not equal to Length mask: {len(mask_paths)}"

            for i in range(len(image_paths)):
                # process slices of images
                if (i == 0):
                    batch = {
                        "left": image_paths[i],
                        "current": image_paths[i],
                        "right": image_paths[i + 1]
                    }
                elif (i == len(image_paths) - 1):
                    batch = {
                        "left": image_paths[i - 1],
                        "current": image_paths[i],
                        "right": image_paths[i]
                    }
                else:
                    batch = {
                        "left": image_paths[i - 1],
                        "current": image_paths[i],
                        "right": image_paths[i + 1]
                    }

                self.images.append(batch)

                # process mask
                self.masks.append(mask_paths[i])

        self.n_samples = len(self.images)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        batch = self.images[index]
        mask = np.load(self.masks[index])

        image_l = cv2.imread(batch["left"], cv2.IMREAD_GRAYSCALE)
        image_r = cv2.imread(batch["right"], cv2.IMREAD_GRAYSCALE)
        image_current = cv2.imread(batch["current"], cv2.IMREAD_GRAYSCALE)

        image_l = image_l / 255.0
        image_l = image_l.astype(np.float32)
        image_l = np.expand_dims(image_l, axis=0)

        image_r = image_r / 255.0
        image_r = image_r.astype(np.float32)
        image_r = np.expand_dims(image_r, axis=0)

        image_current = image_current / 255.0
        image_current = image_current.astype(np.float32)
        image_current = np.expand_dims(image_current, axis=0)

        image = np.stack((image_l, image_current, image_r), axis=0)
        mask = np.expand_dims(mask, 0).astype(np.uint8)

        return image, mask


def load_dataset(total_size, split=0.2):
    split_size = int(total_size * split)
    a = np.arange(0, total_size, 1)
    train, test = train_test_split(a, test_size=split_size)
    print(f"Train: {len(train)} || Test: {len(test)}")
    return train, test


def get_loaders():
    root_images = "../UNETR_MMWHS/files/images/"
    root_masks = "../UNETR_MMWHS/files/masks/"

    train, test = load_dataset(total_size=20)

    train_dataset = ROTATORY_MMWHS(
        root_images=root_images, root_masks=root_masks, num_unique=train)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4)

    val_dataset = ROTATORY_MMWHS(
        root_images=root_images, root_masks=root_masks, num_unique=test)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=4)

    return (train_loader, val_loader)


def main():
    train_loader, val_loader = get_loaders()

    for x, y in train_loader:
        x = torch.squeeze(x, dim=0)
        print(x.shape, y.shape)


if __name__ == "__main__":
    main()
