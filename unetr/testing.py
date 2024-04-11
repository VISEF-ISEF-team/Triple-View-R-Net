import nibabel as nib
import numpy as np
from monai.losses.dice import *
import torch
from monai.losses.dice import DiceLoss
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score


def normalize_image_intensity_range(img):
    HOUNSFIELD_MAX = 4000
    HOUNSFIELD_MIN = 0
    HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX

    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE


def convert_label_to_class(mask):
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


def main():
    image = nib.load(
        "../data_for_training/MMWHS/ct_train/ct_train_1001_image.nii.gz").get_fdata()
    image = normalize_image_intensity_range(image)

    mask = nib.load(
        "../data_for_training/MMWHS/ct_train/ct_train_1001_label.nii.gz").get_fdata()
    convert_label_to_class(mask)
    print(np.min(mask), np.max(mask), np.unique(mask))


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # main()
    B, C, H, W, Z = 1, 8, 128, 128, 128

    # input is model prediction, no need for softmax activation
    input = torch.randn(B, C, H, W, Z)

    # one hot encode label
    target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W, Z)).long()
    target_idx = torch.unsqueeze(target_idx, dim=1)
    # target_idx = target_idx[:, None, ...]
    # target = one_hot(target_idx[:, None, ...], num_classes=C)

    # input = torch.argmax(input, dim=1)

#     print(f"""
#         F1 score: {f1_score(target_idx.flatten(), input.flatten(), average="micro")},
#         Accuracy: {accuracy_score(target_idx.flatten(), input.flatten())},
#         Recall: {recall_score(target_idx.flatten(), input.flatten(), average="micro")},
#         Jaccard: {jaccard_score(target_idx.flatten(), input.flatten(), average="micro")}
# """)

    print(f"Input: {input.shape} || Target: {target_idx.shape}")

    loss_fn = DiceLoss(softmax=True, to_onehot_y=True)
    loss = loss_fn(input, target_idx)

    print(loss.item())
