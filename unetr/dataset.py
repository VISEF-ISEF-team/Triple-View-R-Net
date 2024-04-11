import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from patchify import patchify
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        image = image / 255.0  # (256, 256)
        image = np.expand_dims(image, axis=0)  # (1, 256, 256)
        image = image.astype(np.float32)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask[mask == 255.0] = 1
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples
