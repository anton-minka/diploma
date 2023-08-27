import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    def __init__(
        self,
        log_loss=False,
        from_logits=True,
        smooth: float = 0.0,
        ignore_index=None,
        eps=1e-7,
    ):
        super(DiceLoss, self).__init__()

        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: N,C,H,W
        :param y_true: N,H,W
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = y_pred.log_softmax(dim=1).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, -1)  # N,H*W
        y_pred = y_pred.view(bs, num_classes, -1)  # N,C,H,W -> N,C,H*W

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask.unsqueeze(1)  # N, C, H*W

            y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
        else:
            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)  # C

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return loss.mean()


def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    """
    :param output: N,C,HxW
    :param target: N,C,H*W
    :param smooth:
    :param eps:
    :return:
    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp(min=eps)
    return dice_score