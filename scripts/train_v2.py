import sys
import os
cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(cwd)

import argparse
from collections import defaultdict
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from datasets.ade20k import ADE10KDataset as Dataset
from models import create_model
from utils.augmentation import Augmentations, RandomHorizontallyFlip, RandomRotate
from losses.dice import DiceLoss
from utils.metrics import iou_score, recall, precision, accuracy, dice_coeff
import wandb


def main(args):
    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Unet(n_channels=Dataset.in_channels, n_classes=Dataset.n_classes)
    model = create_model(
        args.model_name or 'unet',
        n_channels=Dataset.in_channels,
        n_classes=Dataset.n_classes
    )
    model.to(device)

    wandb.login(key='fc42bf231508da479c169077f6a27dc7a5eb9401')
    experiment = wandb.init(
        project="diploma",
        config={
            "architecture": model.__class__.__name__.lower(),
            "dataset": "ADE20K",
            "learning_rate": args.lr,
            "epochs": args.epochs,
        },
        dir='../tmp/wandb'
    )
    summary(
        model, input_size=[1, 3, 512, 512], col_names=(
            "input_size",
            "output_size",
            "num_params")
    )

    ce_criteria = nn.CrossEntropyLoss(ignore_index=0)
    dice_criteria = DiceLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score

    best_score = 0
    global_step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for x, y_true in tqdm(loaders[phase], position=0, leave=True, desc=f"Epoch {epoch}"):
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred_logits = model(x)

                    loss = ce_criteria(y_pred_logits, y_true)
                    loss += dice_criteria(y_pred_logits, y_true)

                    epoch_samples += 1
                    global_step += 1

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                        mask_true = F.one_hot(y_true, Dataset.n_classes).permute(0, 3, 1, 2).float()
                        mask_pred = F.one_hot(y_pred_logits.argmax(dim=1), Dataset.n_classes).permute(
                            0, 3, 1, 2
                        ).float()

                        metrics['dice_score'] = dice_coeff(mask_pred, mask_true)
                        metrics['iou'] = iou_score(mask_pred, mask_true)
                        metrics['recall'] = recall(mask_pred, mask_true)
                        metrics['precision'] = precision(mask_pred, mask_true)
                        metrics['accuracy'] = accuracy(mask_pred, mask_true)

                        experiment.log(
                            {
                                'train loss': loss.item(),
                                'train metrics': metrics,
                                'step': global_step,
                                'epoch': epoch
                            }
                        )

                        scheduler.step(metrics['dice_score'])

                    if phase == "valid":
                        mask_true = F.one_hot(y_true, Dataset.n_classes).permute(0, 3, 1, 2).float()
                        mask_pred = F.one_hot(y_pred_logits.argmax(dim=1), Dataset.n_classes).permute(
                            0, 3, 1, 2
                        ).float()
                        metrics['dice_score'] += dice_coeff(mask_pred, mask_true)
                        metrics['iou'] += iou_score(mask_pred, mask_true)
                        metrics['accuracy'] += accuracy(mask_pred, mask_true)
                        metrics['precision'] += precision(mask_pred, mask_true)
                        metrics['recall'] += recall(mask_pred, mask_true)

            if phase == "valid":
                metrics['dice_score'] /= epoch_samples
                metrics['iou'] /= epoch_samples
                metrics['recall'] /= epoch_samples
                metrics['precision'] /= epoch_samples
                metrics['accuracy'] /= epoch_samples

                print('Validation Dice score: {}'.format(metrics['dice_score']))

                experiment.log(
                    {
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation metrics': metrics,
                        'images': wandb.Image(x[0].cpu()),
                        'masks': {
                            'true': wandb.Image(y_true[0].float().cpu()),
                            'pred': wandb.Image(y_pred_logits.argmax(dim=1)[0].float().cpu()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                    }
                )

                if metrics['dice_score'] > best_score:
                    print("saving best model")
                    best_score = metrics['dice_score']

                    if args.checkpoints:
                        path = "../data/checkpoint{}_{}_{}x{}_{}ep{}_{}.pth".format(
                            epoch,
                            model.__class__.__name__.lower(),
                            args.image_size,
                            args.image_size,
                            args.epochs,
                            '_aug' if args.aug else '',
                            datetime.now().strftime('%d-%m-%y_%H%M')
                        )
                        torch.save(model, path)

    print('Best val loss: {:4f}'.format(best_score))

    path = "../data/{}_{}x{}_{}ep{}_{}.pth".format(
        model.__class__.__name__.lower(),
        args.image_size,
        args.image_size,
        args.epochs,
        '_aug' if args.aug else '',
        datetime.now().strftime('%d-%m-%y_%H%M')
    )
    torch.save(model, path)


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size)

    return loader_train, loader_valid


def datasets(args):
    augments = None
    if args.aug:
        augments = Augmentations([RandomHorizontallyFlip(0.9), RandomRotate(degrees=10)])

    train = Dataset('../data/', mode='train', full_dataset=args.full_ds, base_size=args.image_size, augmentations=augments)
    valid = Dataset('../data/', mode='validation', full_dataset=args.full_ds, base_size=args.image_size)
    return train, valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="input batch size for training (default: 16)")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("-s", "--image-size", type=int, default=256, help="target input image size (default: 256)")
    parser.add_argument("-n", "--model-name", default='unet', help="model name (default: unet)")
    parser.add_argument("-l", "--lr", type=float, default=0.001, help="initial learning rate (default: 0.001)")
    parser.add_argument("-a", "--aug", action="store_true", help="use augmentation (default: False)")
    parser.add_argument("-c", "--checkpoints", action="store_true", help="save checkpoints with the best val score (default: False)")
    parser.add_argument("-f", "--full-ds", action="store_true", help="use full dataset (default: False)")
    args = parser.parse_args()
    main(args)
