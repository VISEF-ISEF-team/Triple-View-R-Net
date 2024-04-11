import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from unet2D import Unet
from mmdataset2D import get_loaders
import cv2
import nibabel as nib
from skimage.transform import resize
import torchvision


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
    device = torch.device("cuda")
    model = Unet(inc=1, outc=8)
    model.load_state_dict(torch.load(
        "./files/2D_unet_checkpoint.pth.tar"))
    model = model.to(device)

    train_loader, val_loader = get_loaders()

    save_predictions_as_imgs(val_loader, model)

    # a = cv2.imread("./files/images/heart0-slice309_axial.png",
    #                cv2.IMREAD_GRAYSCALE) / 255.0
    # a = np.expand_dims(a, axis=0)
    # a = np.expand_dims(a, axis=0)
    # a = a.astype(np.float32)
    # a = torch.from_numpy(a).to(device)

    # mask = np.load("./files/masks/heartmaskencode0-slice309_axial.npy")

    # output = torch.argmax(model(a), dim=1)

    # # convert from cuda to cpu
    # output = output.detach().cpu().numpy()[0]

    # a = a.detach().cpu().numpy()
    # a = np.squeeze(a)

    # print(output.shape)
    # print(a.shape)

    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 16))

    # axes[0].imshow(a)
    # axes[0].set_xlabel("IMAGE")
    # axes[1].imshow(output)
    # axes[1].set_xlabel("PRED")
    # axes[2].imshow(mask)
    # axes[2].set_xlabel("MASK")

    # plt.show()
