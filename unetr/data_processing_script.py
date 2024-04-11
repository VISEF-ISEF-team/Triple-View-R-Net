import nibabel as nib
import numpy as np
import os
import nibabel.orientations as ornt
import matplotlib.pyplot as plt
import cv2
from glob import glob

global IMG_W
global IMG_H
global IMG_Z


def normalize_image_intensity_range(img):
    # HOUNSFIELD UNIT ALREADY PROCESSED
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def save_image(img_path: str, nii_index: int, view: str = "axial"):
    array = nib.load(img_path).get_fdata()
    array = normalize_image_intensity_range(array)

    if view == "axial":
        for i in range(array.shape[-1]):
            img_slice = array[:, :, i]
            img_name = f"synapse{nii_index}-slice{str(i).zfill(3)}_axial"
            img = cv2.resize(img_slice, (IMG_W, IMG_H))
            img = np.uint8(img * 255)

            path = os.path.join("../data_for_training",
                                "images", f"{img_name}.png")

            print(f"Image name: {img_name}")

            res = cv2.imwrite(path, img)
            if not res:
                print(f"Error, unable to save image with name: {img_name}")

    elif view == "saggital":
        for i in range(array.shape[1]):
            img_slice = array[:, i, :]
            img_name = f"synapse{nii_index}-slice{str(i).zfill(3)}_saggital"
            img = cv2.resize(img_slice, (IMG_W, IMG_Z))
            img = np.uint8(img_slice * 255)

            path = os.path.join("../data_for_training",
                                "images", f"{img_name}.png")

            res = cv2.imwrite(path, img)
            if not res:
                print(f"Error, unable to save image with name: {img_name}")

    elif view == "coronal":
        for i in range(array.shape[0]):
            img_slice = array[i, :, :]
            img_name = f"synapse{nii_index}-slice{str(i).zfill(3)}_coronal"
            img = cv2.resize(img_slice, (IMG_H, IMG_Z))
            img = np.uint8(img_slice * 255)

            path = os.path.join("../data_for_training",
                                "images", f"{img_name}.png")

            res = cv2.imwrite(path, img)
            if not res:
                print(f"Error, unable to save image with name: {img_name}")


def save_mask(mask_path: str, nii_index: int, view="axial"):
    output = nib.load(mask_path).get_fdata()

    if view == "axial":
        for i in range(output.shape[2]):

            mask = output[:, :, i]
            mask = cv2.resize(mask, (IMG_W, IMG_H)).astype(np.uint8) * 255

            mask_name = f"synapsemaskencode{nii_index}-slice{str(i).zfill(3)}_axial"
            path = os.path.join("../data_for_training",
                                "masks", f"{mask_name}.png")
            cv2.imwrite(path, mask)

            print(f"Mask name: {mask_name}")

    elif view == "saggital":
        for i in range(output.shape[1]):
            mask = output[:, i, :]
            mask = np.uint(cv2.resize(mask, (IMG_W, IMG_H)))

            mask_name = f"synapsemaskencode{nii_index}-slice{str(i).zfill(3)}_saggital"
            path = os.path.join("../data_for_training",
                                "masks", f"{mask_name}.npy")
            np.save(path, mask)

    elif view == "coronal":
        for i in range(output.shape[0]):

            mask = output[i, :, :]
            mask = np.uint(cv2.resize(mask, (IMG_W, IMG_H)))

            mask_name = f"synapsemaskencode{nii_index}-slice{str(i).zfill(3)}_coronal"
            path = os.path.join("../data_for_training",
                                "masks", f"{mask_name}.npy")
            np.save(path, mask)


def volumetric_to_slice(image_path, mask_path, view="axial", custom_counter: int = None):
    image_sub = sorted(glob(os.path.join(image_path, "*")))
    mask_sub = sorted(glob(os.path.join(mask_path, "*")))

    custom_counter_flag = custom_counter != None
    if view == "axial":
        for nii_index in range(len(image_sub)):
            image_path = image_sub[nii_index]
            mask_path = mask_sub[nii_index]

            if not custom_counter_flag:
                save_mask(mask_path, nii_index, view=view)
                save_image(image_path, nii_index, view=view)
            else:
                save_mask(mask_path, custom_counter, view=view)
                save_image(image_path, custom_counter, view=view)
                custom_counter += 1

    elif view == "saggital":
        for nii_index in range(len(image_sub)):
            image_path = image_sub[nii_index]
            mask_path = mask_sub[nii_index]

            if not custom_counter_flag:
                save_mask(mask_path, nii_index, view=view)
                save_image(image_path, nii_index, view=view)

            else:
                save_mask(mask_path, custom_counter, view=view)
                save_image(image_path, custom_counter, view=view)
                custom_counter += 1

    if view == "coronal":
        for nii_index in range(len(image_sub)):
            image_path = image_sub[nii_index]
            mask_path = mask_sub[nii_index]

            if not custom_counter_flag:
                save_mask(mask_path, nii_index, view=view)
                save_image(image_path, nii_index, view=view)
            else:
                save_mask(mask_path, custom_counter, view=view)
                save_image(image_path, custom_counter, view=view)
                custom_counter += 1


if __name__ == "__main__":
    IMG_W = 256
    IMG_H = 256
    IMG_Z = 256

    root = "../data/"

    image_path = os.path.join(root, "synapse_training")
    mask_path = os.path.join(root, "synapse_training_labels")

    volumetric_to_slice(image_path, mask_path, view="axial")
