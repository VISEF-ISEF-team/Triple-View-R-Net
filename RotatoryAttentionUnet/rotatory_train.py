import os
import time
from glob import glob
import torch
from tqdm import tqdm
import torch.nn as nn
from rotatory_dataset import get_loaders, get_slice_from_volumetric_data, duplicate_end, duplicate_open_end, CustomDiceLoss
from monai.losses.dice import DiceLoss
import datetime
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score
import sys
import csv
from rotatory_attention_unet_v2 import Rotatory_Attention_Unet_v2
from rotatory_attention_unet_v3 import Rotatory_Attention_Unet_v3
from torchvision import transforms


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


def train(model, loader, optimizer, loss_fn, scaler, batch_size, device=torch.device("cuda")):
    model.train()
    train_transform_trivial = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins=5),
    ])

    pbar = tqdm(loader)
    steps = len(loader)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_dice_coef = 0.0
    epoch_jaccard = 0.0
    epoch_recall = 0.0
    epoch_f1 = 0.0
    iter_counter = 0

    for step, (x, y) in enumerate(pbar):
        # x = duplicate_open_end(x)
        # y = duplicate_open_end(y)

        length = x.shape[-1]

        for i in range(0, length, batch_size - 1):
            pbar.set_description(
                f"Iter: {iter_counter} - Step: {step} / {steps}")
            iter_counter += 1

            # ensure balance slice count
            if i + batch_size >= length:
                num_slice = length - i

                if num_slice < 3:
                    for _ in range(3 - num_slice):
                        x = duplicate_end(x)
                        y = duplicate_end(y)

                    num_slice = 3

            else:
                num_slice = batch_size

            x_, y_ = get_slice_from_volumetric_data(
                x, y, i, num_slice, train_transform=train_transform_trivial)
            x_ = x_.to(device)
            y_ = y_.to(device)

            optimizer.zero_grad()

            # forward:
            y_pred = model(x_)
            # y_pred = y_pred[1:-1]
            # y_ = y_[1:-1]

            y_ = nn.functional.one_hot(y_.long(), num_classes=8)
            y_ = torch.squeeze(y_, dim=1)
            y_ = y_.permute(0, 3, 1, 2)

            loss = loss_fn(y_pred, y_)

            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            """Take argmax for accuracy calculation"""
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.detach().cpu().numpy()

            y_ = torch.argmax(y_, dim=1)
            y_ = y_.detach().cpu().numpy()

            """Update batch metrics"""
            batch_accuracy = accuracy_score(
                y_.flatten(), y_pred.flatten())

            batch_jaccard = jaccard_score(
                y_.flatten(), y_pred.flatten(), average="micro")

            batch_recall = recall_score(
                y_.flatten(), y_pred.flatten(), average="micro")

            batch_f1 = f1_score(y_.flatten(),
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

    epoch_loss = epoch_loss / iter_counter
    epoch_dice_coef = 1.0 - epoch_loss
    epoch_accuracy = epoch_accuracy / iter_counter
    epoch_jaccard = epoch_jaccard / iter_counter
    epoch_recall = epoch_recall / iter_counter
    epoch_f1 = epoch_f1 / iter_counter

    return epoch_loss, epoch_dice_coef, epoch_accuracy, epoch_jaccard, epoch_recall, epoch_f1


def evaluate(model, loader, loss_fn, batch_size, device=torch.device("cuda")):

    epoch_loss = 0.0
    epoch_f1 = 0.0
    epoch_accuracy = 0.0
    epoch_recall = 0.0
    epoch_jaccard = 0.0
    epoch_dice_coef = 0.0
    iter_counter = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            # x = duplicate_open_end(x)
            # y = duplicate_open_end(y)

            length = x.shape[-1]

            for i in range(0, length, batch_size - 1):
                iter_counter += 1

                # ensure balance slice count
                if i + batch_size >= length:
                    num_slice = length - i

                    if num_slice < 3:
                        for _ in range(3 - num_slice):
                            x = duplicate_end(x)
                            y = duplicate_end(y)

                        num_slice = 3
                else:
                    num_slice = batch_size

                x_, y_ = get_slice_from_volumetric_data(x, y, i, num_slice)

                x_ = x_.to(device)
                y_ = y_.to(device)

                # pass input through model
                y_pred = model(x_)
                # y_pred = y_pred[1:-1]

                # y_ = y_[1:-1]
                y_ = torch.squeeze(y_, dim=1)
                y_ = nn.functional.one_hot(y_.long(), num_classes=8)
                y_ = y_.permute(0, 3, 1, 2)

                # calculate loss
                loss = loss_fn(y_pred, y_)
                epoch_loss += loss.item()

                # take argmax to calculate other metrices
                y_pred = torch.argmax(y_pred, dim=1)
                y_ = torch.argmax(y_, dim=1)

                # convert to numpy to calculate metrics
                y_pred = y_pred.detach().cpu().numpy()
                y_ = y_.detach().cpu().numpy()

                # other metrics calculation
                epoch_f1 += f1_score(y_.flatten(),
                                     y_pred.flatten(), average="micro")
                epoch_accuracy += accuracy_score(
                    y_.flatten(), y_pred.flatten())
                epoch_recall += recall_score(
                    y_.flatten(), y_pred.flatten(), average="micro")
                epoch_jaccard += jaccard_score(
                    y_.flatten(), y_pred.flatten(), average="micro")

        epoch_loss = epoch_loss/iter_counter
        epoch_f1 = epoch_f1 / iter_counter
        epoch_accuracy = epoch_accuracy / iter_counter
        epoch_recall = epoch_recall / iter_counter
        epoch_jaccard = epoch_jaccard / iter_counter
        epoch_dice_coef = 1.0 - epoch_loss

    return epoch_loss, epoch_f1, epoch_accuracy, epoch_recall, epoch_jaccard, epoch_dice_coef


def main(load_model=False, starting_epoch=0, starting_lr=1e-3):
    """Define hyper parameters"""
    lr = starting_lr
    batch_size = 4
    device = torch.device("cuda")
    num_epochs = 65
    checkpoint_path = "./files/rotatory_attention_addition_unet_v3_no-resize.pth.tar"
    train_metrics_path = "./files/rotatory_attention_addition__unet_v3_no-resize_train_metrics.csv"
    test_metrics_path = "./files/rotatory_attention_addition_unet_v3_no-resize_test_metrics.csv"

    """Initial write to csv to set rows"""
    if not load_model or starting_epoch == 0:
        write_csv(train_metrics_path, ["Epoch", "LR", "Loss", "Dice",
                                       "Accuracy", "Jaccard", "Recall", "F1"], first=True)

        write_csv(test_metrics_path, ["Epoch", "LR", "Loss", "Dice",
                                      "Accuracy", "Jaccard", "Recall", "F1"], first=True)

    """Initialize model and more"""
    if load_model:
        checkpoint = torch.load(checkpoint_path)
        model = Rotatory_Attention_Unet_v3(image_size=256)
        model.load_state_dict(checkpoint)
        model.to(device)

    else:
        model = Rotatory_Attention_Unet_v3(image_size=256)
        model.to(device)

    """Define loss function"""
    # loss_fn = DiceLoss(to_onehot_y=True, softmax=True,
    #                    include_background=True)

    loss_fn = CustomDiceLoss()

    """Define optimizer and scheduler"""
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=3)

    """Define scalar in case of mixed precision"""
    scaler = torch.cuda.amp.GradScaler()

    """Test model output"""
    r = model(torch.rand(6, 1, 256, 256).to(device))
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
            model, train_loader, optimizer, loss_fn, scaler, batch_size, device)

        # validate function
        valid_loss, valid_f1, valid_accuracy, valid_recall, valid_jaccard, valid_dice_coef = evaluate(
            model, val_loader, loss_fn, batch_size, device)

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
    main(load_model=False, starting_epoch=0, starting_lr=0.5)