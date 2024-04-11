import os
import time
from glob import glob
import torch
from tqdm import tqdm
import torch.nn as nn
from deepsuperivion_dataset import get_loaders
from fct_model import get_fct
import datetime
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score
import pandas as pd
from monai.losses import DiceLoss, TverskyLoss
import torch.nn.functional as F
from monai.metrics import compute_generalized_dice


def seconds_to_hms(seconds):
    time_obj = datetime.timedelta(seconds=seconds)
    return str(time_obj)


def train(model, loader, optimizer, loss_fn, scaler, device=torch.device("cuda")):
    model.train()
    loop = tqdm(loader)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_dice_coef = 0.0

    for step, (x, y) in enumerate(loop):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        # forward
        y_pred = model(x)

        loss1 = loss_fn(y_pred[0], y[0])  # orignal shape
        loss2 = loss_fn(y_pred[1], y[1])
        loss3 = loss_fn(y_pred[2], y[2])

        loss = loss1 + 0.75 * loss2 + 0.5 * loss3

        loss.backward()
        optimizer.step()

        batch_dice_coef = compute_generalized_dice(
            F.one_hot(torch.argmax(y_pred[0], dim=1)), F.one_hot(y[0]).permute(0, 3, 1, 2))

        """Take argmax for accuracy calculation"""
        y_pred = torch.argmax(y_pred[0], dim=1)
        y_pred = y_pred.detach().cpu().numpy()
        y = y[0].detach().cpu().numpy()

        """Update batch metrics"""
        batch_accuracy = accuracy_score(
            y.flatten(), y_pred.flatten())

        batch_loss = loss.item()

        """Update epoch metrics"""
        epoch_loss += batch_loss
        epoch_accuracy += batch_accuracy
        epoch_dice_coef += batch_dice_coef

        """Set loop postfix"""
        loop.set_postfix(
            {"loss": batch_loss, "dice_coef": batch_dice_coef, "accuracy": batch_accuracy})

    epoch_loss = epoch_loss / len(loader)
    epoch_dice_coef = epoch_dice_coef - epoch_loss
    epoch_accuracy = epoch_accuracy / len(loader)

    return epoch_loss, epoch_dice_coef, epoch_accuracy


def evaluate(model, loader, loss_fn, device=torch.device("cuda")):

    epoch_loss = 0.0
    f1 = 0.0
    accuracy = 0.0
    recall = 0.0
    jaccard = 0.0
    dice_coef = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)[0]

            # calculate loss
            loss = loss_fn(y_pred[0], y[0])

            epoch_loss += loss.item()

            dice_coef += compute_generalized_dice(
                F.one_hot(torch.argmax(y_pred, dim=1)), F.one_hot(y[0]).permute(0, 3, 1, 2))

            # take argmax to calculate other metrices
            y_pred = torch.argmax(y_pred, dim=1)

            # convert to numpy to calculate metrics
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            f1 += f1_score(y[0].flatten(),
                           y_pred.flatten(), average="micro")
            accuracy += accuracy_score(
                y[0].flatten(), y_pred.flatten())
            recall += recall_score(
                y[0].flatten(), y_pred.flatten(), average="micro")
            jaccard += jaccard_score(
                y[0].flatten(), y_pred.flatten(), average="micro")

        epoch_loss = epoch_loss/len(loader)
        f1 = f1 / len(loader)
        accuracy = accuracy / len(loader)
        recall = recall / len(loader)
        jaccard = jaccard / len(loader)
        dice_coef = dice_coef / len(loader)

    return epoch_loss, f1, accuracy, recall, jaccard, dice_coef


def main():
    """Define hyper parameters"""
    lr = 1e-3
    device = torch.device("cpu")
    num_epochs = 25
    checkpoint_path = "./files/rotatory_attention_checkpoint.pth.tar"

    """Initialize model and more"""
    model = get_fct(num_class=8)
    model.to(device)

    # loss function
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=3)

    # scaler
    scaler = torch.cuda.amp.GradScaler()

    """Test model output"""
    r = model(torch.rand(3, 1, 256, 256).to(device))
    print(f"Testing model output: {r.shape}")

    """Get loaderes"""
    train_loader, val_loader = get_loaders()

    """ Training the model """
    best_valid_loss = float("inf")

    SCORES = []

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_dice_coef, train_accuracy = train(model, train_loader, optimizer,
                                                            loss_fn, scaler, device)
        valid_loss, f1, accuracy, recall, jaccard, dice_coef = evaluate(
            model, val_loader, loss_fn, device)

        """Append to score list"""
        SCORES.append([train_dice_coef, train_accuracy, train_loss,
                      valid_loss, f1, accuracy, recall, jaccard, dice_coef])

        """Check loss and update learning rate"""
        scheduler.step(valid_loss)

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
        data_str += f'\t LR: {lr} change to {scheduler.get_last_lr()}\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        data_str += f'\t Val. F1: {f1:.3f}\n'
        data_str += f'\t Val. Accuracy: {accuracy:.3f}\n'
        data_str += f'\t Val. Recall: {recall:.3f}\n'
        data_str += f'\t Val. Jaccard: {jaccard:.3f}\n'
        data_str += f'\t Val. Dice Coef: {dice_coef:.3f}\n'
        print(data_str)

        lr = scheduler.get_last_lr()

    df = pd.DataFrame(SCORES, columns=["Train_dice_coef", "Train_accuracy", "Train_loss",
                                       "Val_loss", "Val_f1", "Val_accuracy", "Val_recall", "Val_jaccard", "Val_dice_coef"])

    df.to_csv("./files/rotatory_attention_metrics_record.csv")


if __name__ == "__main__":
    main()
