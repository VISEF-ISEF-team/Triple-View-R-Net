import os
import time
from glob import glob
import torch
from tqdm import tqdm
import torch.nn as nn
from original_attention_dataset import get_loaders
from original_attention_dataset import CustomDiceLoss
from original_unet_attention import Attention_Unet
from monai.losses import DiceLoss, TverskyLoss
import datetime
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score
import pandas as pd
from rotatory_train import write_csv


def seconds_to_hms(seconds):
    time_obj = datetime.timedelta(seconds=seconds)
    return str(time_obj)


def train(model, loader, optimizer, loss_fn, scaler, device=torch.device("cuda")):
    model.train()
    pbar = tqdm(loader)
    total_steps = len(loader)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_dice_coef = 0.0
    epoch_jaccard = 0.0
    epoch_recall = 0.0
    epoch_f1 = 0.0

    for step, (x, y) in enumerate(pbar):
        pbar.set_description(
            f"Train step: {step} / {total_steps}")
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # forward:
        y_pred = model(x)

        y = nn.functional.one_hot(y.long(), num_classes=8)
        y = torch.squeeze(y, dim=1)
        y = y.permute(0, 3, 1, 2)

        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        """Take argmax for accuracy calculation"""
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.detach().cpu().numpy()

        y = torch.argmax(y, dim=1)
        y = y.detach().cpu().numpy()

        """Update batch metrics"""
        batch_accuracy = accuracy_score(
            y.flatten(), y_pred.flatten())

        batch_jaccard = jaccard_score(
            y.flatten(), y_pred.flatten(), average="micro")

        batch_recall = recall_score(
            y.flatten(), y_pred.flatten(), average="micro")

        batch_f1 = f1_score(y.flatten(),
                            y_pred.flatten(), average="micro")

        batch_loss = loss.item()
        batch_dice_coef = 1.0 - batch_loss

        """Update epoch metrics"""
        epoch_loss += batch_loss
        epoch_accuracy += batch_accuracy
        epoch_jaccard += batch_jaccard
        epoch_recall += batch_recall
        epoch_f1 += batch_f1

        """Set loop postfix"""
        pbar.set_postfix(
            {"loss": batch_loss, "dice_coef": batch_dice_coef, "accuracy": batch_accuracy, "iou": batch_jaccard})

    epoch_loss = epoch_loss / total_steps
    epoch_dice_coef = 1.0 - epoch_loss
    epoch_accuracy = epoch_accuracy / total_steps
    epoch_jaccard = epoch_jaccard / total_steps
    epoch_recall = epoch_recall / total_steps
    epoch_f1 = epoch_f1 / total_steps

    return epoch_loss, epoch_dice_coef, epoch_accuracy, epoch_jaccard, epoch_recall, epoch_f1


def evaluate(model, loader, loss_fn, scaler, device=torch.device("cuda")):
    model.eval()
    pbar = tqdm(loader)
    total_steps = len(loader)
    epoch_loss = 0.0
    epoch_f1 = 0.0
    epoch_accuracy = 0.0
    epoch_recall = 0.0
    epoch_jaccard = 0.0
    epoch_dice_coef = 0.0

    with torch.no_grad():
        for step, (x, y) in enumerate(pbar):
            pbar.set_description(f"Validation step: {step} / {total_steps}")
            x = x.to(device)
            y = y.to(device)

            """Forward pass"""
            y_pred = model(x)

            y = torch.squeeze(y, dim=1)
            y = nn.functional.one_hot(y.long(), num_classes=8)
            y = y.permute(0, 3, 1, 2)

            """Calculate loss"""
            loss = loss_fn(y_pred, y)

            """Take argmax to calculate other metrices"""
            y_pred = torch.argmax(y_pred, dim=1)
            y = torch.argmax(y, dim=1)

            """Convert to numpy for metrics calculation"""
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            """Batch metrics calculation"""
            batch_loss = loss.item()
            batch_f1 = f1_score(y.flatten(),
                                y_pred.flatten(), average="micro")
            batch_accuracy = accuracy_score(
                y.flatten(), y_pred.flatten())
            batch_recall = recall_score(
                y.flatten(), y_pred.flatten(), average="micro")
            batch_jaccard = jaccard_score(
                y.flatten(), y_pred.flatten(), average="micro")
            batch_dice_coef = 1.0 - batch_loss

            """Epoch metrics calculation"""
            epoch_loss += batch_loss
            epoch_f1 += batch_f1
            epoch_accuracy += batch_accuracy
            epoch_recall += batch_recall
            epoch_jaccard += batch_jaccard

            pbar.set_postfix(
                {"loss": batch_loss, "dice_coef": batch_dice_coef, "accuracy": batch_accuracy, "iou": batch_jaccard})

    epoch_loss = epoch_loss / total_steps
    epoch_f1 = epoch_f1 / total_steps
    epoch_accuracy = epoch_accuracy / total_steps
    epoch_recall = epoch_recall / total_steps
    epoch_jaccard = epoch_jaccard / total_steps
    epoch_dice_coef = 1.0 - epoch_loss

    return epoch_loss, epoch_f1, epoch_accuracy, epoch_recall, epoch_jaccard, epoch_dice_coef


def main(load_model=False, starting_epoch=0):
    """Define hyper parameters"""
    lr = 1e-3
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_epochs = 25
    checkpoint_path = "./files/original_attention_unet_checkpoint.pth.tar"
    train_metrics_path = "./files/original_atention_unet_train_metrics.csv"
    test_metrics_path = "./files/original_attention_unet_test_metrics.csv"

    """Initial write to csv to set rows"""
    if not load_model or starting_epoch == 0:
        write_csv(train_metrics_path, ["Epoch", "LR", "Loss", "Dice",
                                       "Accuracy", "Jaccard", "Recall", "F1"], first=True)

        write_csv(test_metrics_path, ["Epoch", "LR", "Loss", "Dice",
                                      "Accuracy", "Jaccard", "Recall", "F1"], first=True)

    """Initialize model and more"""
    if load_model:
        checkpoint = torch.load(checkpoint_path)
        model = Attention_Unet()
        model.load_state_dict(checkpoint)
        model.to(device)

    else:
        model = Attention_Unet()
        model.to(device)

    """Initialize model and more"""

    """Define loss function"""
    loss_fn = CustomDiceLoss()

    """Define optimizer and scheduler"""
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=3)

    """Define scalar in case of mixed precision"""
    scaler = torch.cuda.amp.GradScaler()

    """Test model output"""
    r = model(torch.rand(4, 1, 128, 128).to(device))
    print(f"Testing model output: {r.shape}")

    """Get loaderes"""
    train_loader, val_loader = get_loaders()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(starting_epoch, num_epochs, 1):
        start_time = time.time()

        """Main function call"""
        # train function
        train_loss, train_dice_coef, train_accuracy, train_jaccard, train_recall, train_f1 = train(
            model, train_loader, optimizer, loss_fn, scaler, device)

        # validate function
        valid_loss, valid_f1, valid_accuracy, valid_recall, valid_jaccard, valid_dice_coef = evaluate(
            model, val_loader, loss_fn, scaler,  device)

        """WRite to CSV"""
        # write to train
        write_csv(train_metrics_path, [epoch, lr, train_loss, train_dice_coef, train_accuracy,
                  train_jaccard, train_recall, train_f1])

        # write to test
        write_csv(test_metrics_path, [epoch, lr, valid_loss, valid_dice_coef, valid_accuracy,
                  valid_jaccard, valid_recall, valid_f1])

        """Check loss and update learning rate"""
        scheduler.step(round(valid_loss, 4))

        """Saving and checking loss of model"""
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint at: {checkpoint_path}"
            print(data_str)
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        """Calculate total time"""
        end_time = time.time()
        total_seconds = end_time - start_time
        formatted_time = seconds_to_hms(total_seconds)

        """Format string for printing"""
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
    main(load_model=False, starting_epoch=0)
