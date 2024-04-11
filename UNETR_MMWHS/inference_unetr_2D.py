import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from unetr_model import UNETR_2D
from mmunetrdataset2D import get_loaders
import cv2
import nibabel as nib
from skimage.transform import resize
import torchvision
from patchify import patchify


def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def save_predictions_as_imgs(loader, model, folder="./saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = torch.argmax(model(x), dim=1, keepdim=True).float()

        torchvision.utils.save_image(
            preds, f"{folder}/pred_batch_{idx}.png")
        torchvision.utils.save_image(
            y.float(), f"{folder}/mask_batch_{idx}.png")


if __name__ == "__main__":
    """Define hyper parameters"""
    device = torch.device("cuda")
    config = {}
    config["image_size"] = 256
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.5
    config["num_patches"] = 256
    config["patch_size"] = 16
    config["num_channels"] = 1

    model = UNETR_2D(config, outc=8)
    model.load_state_dict(torch.load(
        "./files/2D_unetr_checkpoint.pth.tar"))
    model = model.to(device)

    """Save prediction as images"""
    train_loader, val_loader = get_loaders()
    save_predictions_as_imgs(val_loader, model)

    """Inference on a single image"""
    # process image
    # a = cv2.imread("./files/images/heart0-slice309_axial.png",
    #                cv2.IMREAD_GRAYSCALE) / 255.0
    # patch_shape = (16, 16)
    # a = a.astype(np.float32)
    # patches = patchify(a, patch_size=patch_shape, step=16)
    # patches = np.reshape(patches, (16*16, 16*16))
    # patches = np.expand_dims(patches, axis=0).astype(np.float32)
    # patches = torch.from_numpy(patches).to(device)

    # # process mask
    # mask = np.load("./files/masks/heartmaskencode0-slice309_axial.npy")

    # output = torch.argmax(model(patches), dim=1)[0]
    # output = output.detach().cpu().numpy()

    # print(f"Image shape: {a.shape}")
    # print(f"Output shape: {output.shape}")

    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 16))

    # axes[0].imshow(a)
    # axes[0].set_xlabel("IMAGE")
    # axes[1].imshow(output)
    # axes[1].set_xlabel("PRED")
    # axes[2].imshow(mask)
    # axes[2].set_xlabel("MASK")

    # plt.show()
