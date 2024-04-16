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
import csv


def seconds_to_hms(seconds):
    time_obj = datetime.timedelta(seconds=seconds)
    return str(time_obj)


def write_csv(path, data, first=False):
    if first:
        with open(path, mode='w', newline='') as file:
            iteration = csv.writer(file)
            iteration.writerow(data)
        file.close()

    else:
        with open(path, mode='a', newline='') as file:
            iteration = csv.writer(file)
            iteration.writerow(data)
        file.close()


def train(model, loader, optimizer, loss_fn, deep_weights, scaler, device=torch.device("cuda")):
    model.train()
    loop = tqdm(loader)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_dice_coef = 0.0
    epoch_jaccard = 0.0
    epoch_recall = 0.0
    epoch_f1 = 0.0

    for step, (x, y) in enumerate(loop):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        # forward
        y_pred = model(x)

        loss1 = loss_fn(y_pred[0], y[0])  # orignal shape
        loss2 = loss_fn(y_pred[1], y[1])
        loss3 = loss_fn(y_pred[2], y[2])

        loss = deep_weights[0] * loss1 + \
            deep_weights[1] * loss2 + deep_weights[2] * loss3

        loss.backward()
        optimizer.step()

        """Update batch metrics"""
        for i in range(len(y_pred)):
            y_pred_sub = torch.argmax(y_pred[i], dim=1)
            y_pred_sub = y_pred_sub.detach().cpu().numpy()
            y_sub = y[i].detach().cpu().numpy()

            batch_accuracy += deep_weights[i] * accuracy_score(
                y_sub.flatten(), y_pred_sub.flatten())

            batch_jaccard += deep_weights[i] * jaccard_score(
                y_sub.flatten(), y_pred_sub.flatten(), average="micro")

            batch_recall += deep_weights[i] * recall_score(
                y_sub.flatten(), y_pred_sub.flatten(), average="micro")

            batch_f1 += deep_weights[i] * f1_score(y_sub.flatten(),
                                                   y_pred_sub.flatten(), average="micro")
        batch_loss = loss.item()
        batch_dice_coef = 1.0 - batch_loss

        """Update epoch metrics"""
        epoch_loss += batch_loss
        epoch_accuracy += batch_accuracy
        epoch_jaccard += batch_jaccard
        epoch_recall += batch_recall
        epoch_f1 += batch_f1

        """Set loop postfix"""
        loop.set_postfix(
            {"loss": batch_loss, "dice_coef": batch_dice_coef, "accuracy": batch_accuracy, "iou": batch_jaccard})

    epoch_loss = epoch_loss / len(loader)
    epoch_dice_coef = 1.0 - epoch_loss
    epoch_accuracy = epoch_accuracy / len(loader)
    epoch_jaccard = epoch_jaccard / len(loader)
    epoch_recall = epoch_recall / len(loader)
    epoch_f1 = epoch_f1 / len(loader)

    return epoch_loss, epoch_dice_coef, epoch_accuracy, epoch_jaccard, epoch_recall, epoch_f1


def evaluate(model, loader, loss_fn, deep_weights, scaler, device=torch.device("cuda")):

    epoch_loss = 0.0
    epoch_f1 = 0.0
    epoch_accuracy = 0.0
    epoch_recall = 0.0
    epoch_jaccard = 0.0
    epoch_dice_coef = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            loss1 = loss_fn(y_pred[0], y[0])  # orignal shape
            loss2 = loss_fn(y_pred[1], y[1])
            loss3 = loss_fn(y_pred[2], y[2])

            loss = deep_weights[0] * loss1 + \
                deep_weights[1] * loss2 + deep_weights[2] * loss3

            """Calculate batch metrics"""
            for i in range(len(y_pred)):
                y_pred_sub = torch.argmax(y_pred[i], dim=1)
                y_pred_sub = y_pred_sub.detach().cpu().numpy()
                y_sub = y[i].detach().cpu().numpy()

                batch_accuracy += deep_weights[i] * accuracy_score(
                    y_sub.flatten(), y_pred_sub.flatten())

                batch_jaccard += deep_weights[i] * jaccard_score(
                    y_sub.flatten(), y_pred_sub.flatten(), average="micro")

                batch_recall += deep_weights[i] * recall_score(
                    y_sub.flatten(), y_pred_sub.flatten(), average="micro")

                batch_f1 += deep_weights[i] * f1_score(y_sub.flatten(),
                                                       y_pred_sub.flatten(), average="micro")
            batch_loss = loss.item()

            """Update epoch metrics"""
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            epoch_jaccard += batch_jaccard
            epoch_recall += batch_recall
            epoch_f1 += batch_f1

        epoch_loss = epoch_loss / len(loader)
        epoch_f1 = epoch_f1 / len(loader)
        epoch_accuracy = epoch_accuracy / len(loader)
        epoch_recall = epoch_recall / len(loader)
        epoch_jaccard = epoch_jaccard / len(loader)
        epoch_dice_coef = 1.0 - epoch_loss

    return epoch_loss, epoch_f1, epoch_accuracy, epoch_recall, epoch_jaccard, epoch_dice_coef


def main():
    """Define hyper parameters"""
    lr = 1e-3
    device = torch.device("cpu")
    num_epochs = 65
    checkpoint_path = "./files/rotatory_attention_checkpoint.pth.tar"
    train_metrics_path = "./files/rotatory_attention_train_metrics.csv"
    test_metrics_path = "./files/rotatory_attention_test_metrics.csv"
    deep_weights = [0.5, 0.3, 0.2]

    """Initial write to csv to set rows"""
    write_csv(train_metrics_path, ["Loss", "Dice",
              "Accuracy", "Jaccard", "Recall", "F1"], first=True)

    write_csv(test_metrics_path, ["Loss", "Dice",
              "Accuracy", "Jaccard", "Recall", "F1"], first=True)

    """Initialize model and more"""
    model = get_fct(num_class=8)
    model.to(device)

    """Define loss function"""
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)

    """Define optimizer and scheduler"""
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=3)

    """Define scalar in case of mixed precision"""
    scaler = torch.cuda.amp.GradScaler()

    """Test model output"""
    r = model(torch.rand(3, 1, 256, 256).to(device))
    print(f"Testing model output: {r.shape}")

    """Get loaderes"""
    train_loader, val_loader = get_loaders()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        # train function
        train_loss, train_dice_coef, train_accuracy, train_jaccard, train_recall, train_f1 = train(
            model, train_loader, optimizer, loss_fn, deep_weights, scaler, device)

        # validate function
        valid_loss, valid_f1, valid_accuracy, valid_recall, valid_jaccard, valid_dice_coef = evaluate(
            model, val_loader, loss_fn, deep_weights, scaler, device)

        """Write to csv"""
        # write to train
        write_csv(train_metrics_path, [train_loss, train_dice_coef, train_accuracy,
                  train_jaccard, train_recall, train_f1])

        # write to test
        write_csv(test_metrics_path, [valid_loss, valid_dice_coef, valid_accuracy,
                  valid_jaccard, valid_recall, valid_f1])

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
        data_str += f'\t Val. F1: {valid_f1:.3f}\n'
        data_str += f'\t Val. Accuracy: {valid_accuracy:.3f}\n'
        data_str += f'\t Val. Recall: {valid_recall:.3f}\n'
        data_str += f'\t Val. Jaccard: {valid_jaccard:.3f}\n'
        data_str += f'\t Val. Dice Coef: {valid_dice_coef:.3f}\n'
        print(data_str)

        """Update lr value"""
        lr = scheduler.get_last_lr()


if __name__ == "__main__":
    main()
