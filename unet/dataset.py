import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from glob import glob


class SynapseDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = cv2.imread(
            self.image_path[index], cv2.IMREAD_GRAYSCALE)
        image = image / 255
        image = np.expand_dims(image, axis=0).astype(np.float32)

        mask = cv2.imread(self.mask_path[index], cv2.IMREAD_GRAYSCALE)
        mask[mask == 255.0] = 1
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        return image, mask
