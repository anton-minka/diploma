
import os

import cv2
import numpy as np

import torch
from torch.utils import data
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import matplotlib.pyplot as plt
from utils.augmentation import Augmentations, RandomHorizontallyFlip, RandomRotate


class TestDataset(Dataset):
    in_channels = 3

    def __init__(self, root, mode='test', base_size=512):
        self.root = root
        self.mode = mode
        self.base_size = base_size

        self.base_dir_path = os.path.join(root, "ADEChallengeData2016", "test")
        assert os.path.exists(self.base_dir_path), "Dataset does not exist"

        self.images = []
        for image_file_name in os.listdir(self.base_dir_path):
            image_file_stem, _ = os.path.splitext(image_file_name)
            if image_file_name.endswith(".jpg"):
                image_file_path = os.path.join(self.base_dir_path, image_file_name)
                self.images.append(image_file_path)

        assert len(self.images) > 0, "Not found any image file"

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")

        image = image.resize((self.base_size, self.base_size), Image.BILINEAR)

        return self.img_transform(image)

    def __len__(self):
        return len(self.images)

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy())
        return img


if __name__ == "__main__":
    data = TestDataset("../data/")
    for img in data:
        print(img.shape)
        img = np.transpose(img.numpy(), (1, 2, 0))
        img = img[:, :, ::-1]
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

