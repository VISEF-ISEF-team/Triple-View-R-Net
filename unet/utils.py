import torch
import torchvision
from dataset import SynapseDataset
from torch.utils.data import DataLoader
import os
from glob import glob
from sklearn.model_selection import train_test_split


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to: {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print(f"=> Load checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def load_dataset(image_path, mask_path, split=0.2):
    images = sorted(glob(os.path.join(image_path, "*.png")))
    masks = sorted(glob(os.path.join(mask_path, "*.png")))

    split_size = int(len(images) * split)

    x_train, x_val = train_test_split(
        images, test_size=split_size, random_state=42)
    y_train, y_val = train_test_split(
        masks, test_size=split_size, random_state=42)

    return (x_train, y_train), (x_val, y_val)


def get_loaders(batch_size, train_transform=None, val_transform=None, num_workers=4, pin_memory=True):
    image_path = os.path.join("../data_for_training/images/")
    mask_path = os.path.join("../data_for_training/masks/")

    (x_train, y_train), (x_val, y_val) = load_dataset(image_path, mask_path)

    train_ds = SynapseDataset(x_train, y_train, transform=None)
    val_ds = SynapseDataset(x_val, y_val, transform=None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / \
                ((preds.sum() + y.sum()) + 1e-8)

    print(
        f"Got {num_correct} / {num_pixels} with acc {(num_correct/num_pixels) *100 :.2f}")
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(epoch, loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_epoch{epoch}_batch_{idx}.png")
        torchvision.utils.save_image(
            y.float(), f"{folder}/mask_epoch{epoch}_batch_{idx}.png")
    model.train()
