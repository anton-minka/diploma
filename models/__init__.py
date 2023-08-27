from .unet import Unet
from .fcn import FCN8s, VGGNet
import segmentation_models_pytorch as smp


def create_model(
    name,
    n_channels: int = 3,
    n_classes: int = 1,
):
    name = name.lower()
    if name == 'unet':
        return Unet(n_channels=n_channels, n_classes=n_classes)
    if name == 'deeplabv3':
        return smp.DeepLabV3(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=n_channels,
            classes=n_classes,
        )
    if name == 'fpn':
        return smp.FPN(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=n_channels,
            classes=n_classes,
        )
    if name == 'fcn':
        return FCN8s(n_classes=n_classes, encoder=VGGNet(pretrained=True))

    raise ValueError(f"Not found model name {name}")

