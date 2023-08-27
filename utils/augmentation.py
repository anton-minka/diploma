import random
from PIL import Image
import torchvision.transforms.functional as transforms
from torchvision.transforms import RandomRotation


class Augmentations(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        return img, mask


class RandomHorizontallyFlip(object):
    def __init__(self, r=0.5):
        self.r = r

    def __call__(self, img, mask):
        if random.random() < self.r:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask


class RandomRotate(object):
    def __init__(self, degrees=45):
        self.degrees = random.random() * 2 * degrees - degrees

    def __call__(self, img, mask):
        return (
            transforms.affine(
                img,
                translate=[0, 0],
                scale=1.0,
                angle=self.degrees,
                interpolation=Image.BILINEAR,
                fill=255,
                shear=0.0,
            ),
            transforms.affine(
                mask,
                translate=[0, 0],
                scale=1.0,
                angle=self.degrees,
                interpolation=Image.NEAREST,
                fill=0,
                shear=0.0,
            )
        )
