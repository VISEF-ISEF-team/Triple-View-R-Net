import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from unet3d import Unet3D
import cv2
import nibabel as nib
from skimage.transform import resize


def normalize_image_intensity_range(img):
    HOUNSFIELD_MAX = 4000
    HOUNSFIELD_MIN = 0
    HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX

    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE


def one_hot_encode(mask):
    output = np.stack([np.where(mask == x, 1, 0)
                      for x in np.unique(mask)], axis=0)
    return output


def read_image(image_path):
    image = nib.load(image_path).get_fdata()

    image = normalize_image_intensity_range(image)
    reshaped_image = resize(
        image, (128, 128, 128), anti_aliasing=True, preserve_range=True, order=1)
    reshaped_image = np.expand_dims(reshaped_image, axis=0)
    reshaped_image = np.round(reshaped_image, 2)
    reshaped_image = reshaped_image.astype(np.float32)

    return reshaped_image


if __name__ == "__main__":
    device = torch.device("cuda")
    model = Unet3D(inc=1, outc=8)
    model.load_state_dict(torch.load(
        "./files/checkpoint.pth.tar"))
    model = model.to(device)

    a = read_image(
        "../data_for_training/ImageCHD_3D/res_images_512/image_0001.nii.gz")

    print(np.min(a), np.max(a), a.shape)

    a = np.expand_dims(a, axis=0)
    a = torch.from_numpy(a).to(device)

    output = model(a)[0]
    output = torch.argmax(output, dim=0)
    output = output.cpu().numpy().astype(np.int32)

    print(output.shape)
    print(np.unique(output))

    i = 0
    for x in np.unique(output):
        inner_output = np.where(output == x, 1, 0)
        nifti_image = nib.nifti1.Nifti1Image(
            inner_output, affine=np.eye(4), dtype=np.int32)
        nib.save(nifti_image, f"./predictions_{i}.nii")
        i += 1
