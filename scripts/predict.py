import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from datasets.test import TestDataset
from models.unet import Unet

DESIRED_LABELS = (4, 7, 12, 14, 29, 53, 55, 92, 95)
ROOT_PATH = '../data'

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_file_name = "unet_256x256_50_ep_21-08-2023_1913.pth"
    file_path = os.path.join(ROOT_PATH, model_file_name)

    # model = torch.load(file_path)
    # model.to(device)

    model = Unet(n_channels=3, n_classes=151)
    model.load_state_dict(torch.load(file_path))
    model.to(device)


    data = TestDataset("../data/")
    for img in data:
        mask = predict(model, img, device)
        img = np.transpose(img.numpy(), (1, 2, 0))

        # fig, arr = plt.subplots(1, 2, figsize=(14, 10))
        # arr[0].imshow(img)
        # arr[0].set_title('Image')
        # arr[1].imshow(mask, cmap='Paired')
        # arr[1].set_title('Segmentation')

        mask[mask == 0] = 'nan'
        plt.imshow(img)
        plt.imshow(mask, 'gist_rainbow', alpha=0.5)
        plt.show()


def predict(model, image, device):
    x = image.to(device)
    output = model(x.float().unsqueeze(0)).cpu()

    mask = output.argmax(dim=1)
    mask = mask[0].long().squeeze().numpy()

    coords = np.argwhere(np.isin(mask, DESIRED_LABELS)).T
    ground = np.zeros((mask.shape[0], mask.shape[1]))
    ground[coords[0], coords[1]] = 1

    return ground


if __name__ == '__main__':
    main()