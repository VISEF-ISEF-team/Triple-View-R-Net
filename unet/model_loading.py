import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import Unet
import cv2
from utils import get_loaders

if __name__ == "__main__":
    # device = torch.device("cuda")
    # model = Unet(1, 1)
    # model.load_state_dict(torch.load(
    #     "./my_checkpoint.pth.tar")["state_dict"])
    # model = model.to(device)

    # image = cv2.imread(
    #     "../data_for_training/images/synapse0-slice006_axial.png", cv2.IMREAD_GRAYSCALE)
    # image = image / 255
    # image = np.expand_dims(image, axis=0)
    # image = np.expand_dims(image, axis=0).astype(np.float32)
    # image = torch.from_numpy(image)
    # image = image.to(device)
    # print(f"Image shape: {image.shape}")

    # mask = cv2.imread("../data_for_training/masks/synapsemaskencode0-slice006_axial.png",
    #                   cv2.IMREAD_GRAYSCALE)
    # mask[mask == 255.0] = 1
    # mask = torch.from_numpy(mask)
    # mask = mask.to(device)
    # print(f"Mask shape: {mask.shape}")

    # preds = model(image)

    # print(f"Preds shape: {preds.shape}")

    train_loader, val_loader = get_loaders(
        batch_size=2, train_transform=None, val_transform=None, num_workers=4, pin_memory=True)

    for x, y in train_loader:
        y_numpy = y.numpy()

        print(y_numpy.shape)
