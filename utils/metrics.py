import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def dice_coeff(input, target, eps=1e-6):
    input = input.flatten(0, 1)
    target = target.flatten(0, 1)

    assert input.size() == target.size()

    sum_dim = (-1, -2)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + eps) / (sets_sum + eps)
    return dice.mean()


def accuracy(y_pred, y_true):
    input = y_pred.flatten(0, 1)
    target = y_true.flatten(0, 1)

    assert input.size() == target.size()

    tp = torch.sum(target == input, dtype=input.dtype)
    score = tp / target.view(-1).shape[0]
    return score.mean()


def iou_score(y_pred, y_true, eps=1e-7):
    input = y_pred.flatten(0, 1)
    target = y_true.flatten(0, 1)

    assert input.size() == target.size()

    sum_dim = (-1, -2)

    inter = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - inter

    score = (inter + eps) / (union + eps)
    return score.mean()


def precision(y_pred, y_true, eps=1e-7):
    input = y_pred.flatten(0, 1)
    target = y_true.flatten(0, 1)

    assert input.size() == target.size()

    sum_dim = (-1, -2)

    tp = (input * target).sum(dim=sum_dim)
    fp = input.sum(dim=sum_dim) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score.mean()


def recall(y_pred, y_true, eps=1e-7):
    input = y_pred.flatten(0, 1)
    target = y_true.flatten(0, 1)

    assert input.size() == target.size()

    sum_dim = (-1, -2)

    tp = (input * target).sum(dim=sum_dim)
    fn = target.sum(dim=sum_dim) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score.mean()


def conf_matrix(
    output: torch.LongTensor,  # N, C, H, W
    target: torch.LongTensor,  # N, H, W
):
    batch_size, num_classes, *dims = output.shape

    output = output.argmax(dim=1)  # N, H, W

    output = output.view(batch_size, -1)  # N,H*W
    target = target.view(batch_size, -1)  # N,H*W

    output = F.one_hot(output, num_classes).permute(0, 2, 1)  # N, C, H*W
    target = F.one_hot(target, num_classes).permute(0, 2, 1)  # N, C, H*W

    tp = (output * target).sum(2)
    fp = output.sum(2) - tp
    fn = target.sum(2) - tp
    tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

    return tp, fp, fn, tn


def __f1_score(tp, fp, fn, tn, eps=1e-7):
    _pr = tp / (tp + fp + eps)
    _rc = tp / (tp + fn + eps)
    return 2 * _pr * _rc / (_pr + _rc + eps)


def __iou_score(tp, fp, fn, tn, eps=1e-7):
    return tp / (tp + fp + fn + eps)


def __accuracy(tp, fp, fn, tn, eps=1e-7):
    return (tp + tn + eps) / (tp + fp + fn + tn + eps)


def __recall(tp, fp, fn, tn, eps=1e-7):
    return tp / (tp + fn + eps)


def __precision(tp, fp, fn, tn, eps=1e-7):
    return tp / (tp + fp + eps)



if __name__ == '__main__':
    mask = Image.open('../data/ADEChallengeData2016/full/annotations/training/ADE_train_00000034.png')
    mask = mask.resize((256, 256), Image.NEAREST)
    mask = torch.from_numpy(np.array([mask])).long()
    print(mask.shape)

    mask_true = F.one_hot(mask, 151).permute(0, 3, 1, 2).float()
    score = iou_score(mask_true, mask_true)
    print(score)
