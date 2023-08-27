import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter

from torch.utils import data


class ADE20KLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 151
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)

        if not self.test_mode:
            for split in ["training", "validation"]:
                rootdir = self.root + "images\\ADE\\" + self.split + "\\"
                suffix = ".jpg"
                file_list = [
                    os.path.join(looproot, filename)
                    for looproot, _, filenames in os.walk(rootdir)
                    for filename in filenames
                    if filename.endswith(suffix)
                ]
                self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path[:-4] + "_seg.png"

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_COLOR)
        print('------------------------')

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        plt.imshow(cv2.cvtColor(np.transpose(img.numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
        plt.show()
        return img, lbl

    def transform(self, img, lbl):
        # img in BGR mode
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img.astype(np.float64)

        # img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        lbl = self.encode_segmap(lbl)
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl))

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def encode_segmap(self, mask):
        # Refer : https://github.com/CSAILVision/ADE20K/blob/main/utils/utils_ade20k.py
        B = mask[:, :, 0]
        G = mask[:, :, 1]
        R = mask[:, :, 2]

        label_mask = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))
        return np.array(label_mask, dtype=np.uint8)

    def decode_segmap(self, temp, plot=False):
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = 10 * (l % 10)
            g[temp == l] = l
            b[temp == l] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


if __name__ == "__main__":
    local_path = "../data/ADE20K_2021_17_01/"
    dst = ADE20KLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    # img, mask = dst[299]
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.imshow(dst.decode_segmap(mask.numpy()))
    # plt.show()
    #
    # print(mask, dst.decode_segmap(mask.numpy()))

    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            # img = img[:, :, ::-1]
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
            for j in range(4):
                print(labels.numpy()[j])
                plt.imshow(dst.decode_segmap(labels.numpy()[j]))
                plt.show()

