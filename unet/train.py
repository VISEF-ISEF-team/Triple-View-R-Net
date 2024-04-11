import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import Unet
from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs
from loss import DiceLoss

"""Hyperparameters"""
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_indx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    model = Unet(inc=1, outc=1).to(device=DEVICE)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        batch_size=BATCH_SIZE, train_transform=None, val_transform=None, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # save checkpoint
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # save images to folder
        save_predictions_as_imgs(epoch, val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()
