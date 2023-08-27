
import os

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.augmentation import Augmentations, RandomHorizontallyFlip, RandomRotate


class ADE10KDataset(Dataset):
    in_channels = 3
    n_classes = 151

    def __init__(self, root, transform=None, mode='train', base_size=512, full_dataset=True, augmentations=None):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.base_size = base_size
        self.augmentations = augmentations

        self.base_dir_path = os.path.join(root, "ADEChallengeData2016", "full" if full_dataset else "short")
        assert os.path.exists(self.base_dir_path), "Dataset does not exist"

        image_dir_path = os.path.join(self.base_dir_path, "images")
        mask_dir_path = os.path.join(self.base_dir_path, "annotations")

        mode_dir_name = "training" if mode == "train" else "validation"
        image_dir_path = os.path.join(image_dir_path, mode_dir_name)
        mask_dir_path = os.path.join(mask_dir_path, mode_dir_name)

        self.images = []
        self.masks = []
        for image_file_name in os.listdir(image_dir_path):
            image_file_stem, _ = os.path.splitext(image_file_name)
            if image_file_name.endswith(".jpg"):
                image_file_path = os.path.join(image_dir_path, image_file_name)
                mask_file_name = image_file_stem + ".png"
                mask_file_path = os.path.join(mask_dir_path, mask_file_name)
                if os.path.isfile(mask_file_path):
                    self.images.append(image_file_path)
                    self.masks.append(mask_file_path)
                else:
                    print("Cannot find the mask: {}".format(mask_file_path))

        assert len(self.images) > 0, "Not found any image file"
        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        assert(mask.mode == "L")
        assert (image.size[0] == mask.size[0])
        assert (image.size[1] == mask.size[1])
        # print(np.array(image).shape, np.array(mask).shape)

        # print(mask, self.encode_segmap(np.array(mask, dtype=np.int32)))

        image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
        mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        # print(np.array(image).shape, np.array(mask).shape)

        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)

        return self.img_transform(image), self.segm_transform(mask)

    def __len__(self):
        return len(self.images)

    def get_labels(self):
        path = os.path.join(self.base_dir_path, "objectInfo150.txt")
        assert os.path.exists(path), 'Labels file {} not exists'.format(path)

        with open(path, "r") as f:
            lines = f.readlines()[1:]

        labels = {}
        for line in lines:
            _id, _, _, _, _labels = line.strip().split("\t")
            _label = _labels.split(",")[0]  # just use first word for each id.
            labels[_label] = int(_id)

        # id 0 is unknown class on ade20k dataset
        assert 0 not in labels
        labels['UNKNOWN'] = 0

        return labels

    @staticmethod
    def _img_transform(image):
        return np.array(image).transpose((2, 0, 1))

    @staticmethod
    def _mask_transform(mask):
        return np.array(mask).astype(np.int32)

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        # print(img.shape)
        img = img.transpose((2, 0, 1))
        # print(img.shape)
        img = torch.from_numpy(img.copy())
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        # segm = torch.from_numpy(np.array(segm)).long() - 1
        segm = torch.from_numpy(np.array(segm)).long()
        return segm


if __name__ == "__main__":
    local_path = "../data/"

    augments = Augmentations([RandomHorizontallyFlip(0.9), RandomRotate(degrees=10)])
    dst = ADE10KDataset(local_path, full_dataset=False, augmentations=augments)
    trainloader = data.DataLoader(dst, batch_size=4)
    # img, mask = dst[3]
    # print(img.shape, mask.shape)
    # plt.imshow(cv2.cvtColor(img.numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.imshow(mask)
    # plt.show()
    #
    # print(mask)

    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
            for j in range(4):
                print(labels.numpy()[j])
                plt.imshow(labels.numpy()[j])
                plt.show()

