import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets.ade20k import ADE10KDataset as Dataset
from torch.utils.data import DataLoader

from utils.metrics import dice_coeff


def evaluate(model, dataloader, device, amp=False):
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = model(image)

            assert mask_true.min() >= 0 and mask_true.max() < model.n_classes, 'True mask indices should be in [0, n_classes]'
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()

            dice_score += dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    model.train()
    return dice_score / max(num_val_batches, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("-m", "--model-name", type=str, help="model name")
    parser.add_argument("-s", "--image-size", type=int, default=256, help="image size (default: 256)")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size (default: 8)")
    args = parser.parse_args()

    dataset = Dataset('../data/', mode='validation', full_dataset=True, base_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = "../data/{}.pth".format(args.model_name)
    model = torch.load(path)

    score = evaluate(model, dataloader, device)
    print('Validation Dice score: {}'.format(score))

