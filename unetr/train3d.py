import os
import time
from glob import glob
import torch
from tqdm import tqdm
import torch.nn as nn
from chddataset3d import get_loaders
from monai.losses import DiceLoss, TverskyLoss
from unetr3d import UneTR
from unet3d import Unet3D
import torch.optim as opt
import datetime
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score


def seconds_to_hms(seconds):
    time_obj = datetime.timedelta(seconds=seconds)
    return str(time_obj)


def train(model, loader, optimizer, loss_fn, scaler, device=torch.device("cuda")):
    loop = tqdm(loader)
    epoch_loss = 0.0
    model.train()

    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        # forward
        with torch.cuda.amp.autocast():
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device=torch.device("cuda")):
    epoch_loss = 0.0
    f1 = 0.0
    accuracy = 0.0
    recall = 0.0
    jaccard = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            # calculate loss
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            # take argmax to calculate other metrices
            y_pred = torch.argmax(y_pred, dim=1)

            # convert to numpy to calculate metrics
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            f1 += f1_score(y.flatten(),
                           y_pred.flatten(), average="micro")
            accuracy += accuracy_score(
                y.flatten(), y_pred.flatten())
            recall += recall_score(
                y.flatten(), y_pred.flatten(), average="micro")
            jaccard += jaccard_score(
                y.flatten(), y_pred.flatten(), average="micro")

        epoch_loss = epoch_loss/len(loader)
        f1 = f1 / len(loader)
        accuracy = accuracy / len(loader)
        recall = recall / len(loader)
        jaccard = jaccard / len(loader)

    return epoch_loss, f1, accuracy, recall, jaccard


def main():
    """Define hyper parameters"""
    lr = 1e-4
    device = torch.device("cuda")
    num_epochs = 50
    checkpoint_path = "./files/checkpoint.pth.tar"

    """Initialize model and more"""
    # model = UneTR(img_dim=(256, 256, 16),
    #               in_channels=1,
    #               base_filter=16,
    #               class_num=1,
    #               patch_size=16,
    #               embedding_dim=768,
    #               block_num=12,
    #               head_num=12,
    #               mlp_dim=3072,
    #               z_idx_list=[3, 6, 9, 12])

    model = Unet3D(inc=1, outc=8)
    model.to(device)

    # loss functoin
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    """Test model output"""
    r = model(torch.rand(1, 1, 128, 128, 128).to(device))
    print(f"Testing model output: {r.shape}")

    """Get loaderes"""
    train_loader, val_loader = get_loaders()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer,
                           loss_fn, scaler, device)
        valid_loss, f1, accuracy, recall, jaccard = evaluate(
            model, val_loader, loss_fn, device)

        """Saving and checking loss of model"""
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint at: {checkpoint_path}"
            print(data_str)
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        total_seconds = end_time - start_time
        formatted_time = seconds_to_hms(total_seconds)
        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {formatted_time}\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        data_str += f'\t Val. F1: {f1:.3f}\n'
        data_str += f'\t Val. Accuracy: {accuracy:.3f}\n'
        data_str += f'\t Val. Recall: {recall:.3f}\n'
        data_str += f'\t Val. Jaccard: {jaccard:.3f}\n'
        print(data_str)


if __name__ == "__main__":
    main()
