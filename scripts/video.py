import argparse
import os
import sys

cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(cwd)

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from models.unet import Unet
from models.fcn import FCN16s
from utils.color import create_colormap, visualize_semantic

DESIRED_LABELS = (4, 7, 12, 14, 29, 53, 55, 92, 95)
ROOT_PATH = '../data'


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_file_name = args.file_name if args.file_name else "fpn_256x256_10ep_aug_23-08-23_1236.pth"
    file_path = os.path.join(ROOT_PATH, model_file_name)
    try:
        model = Unet(n_channels=3, n_classes=151)
        state_dict = torch.load(file_path)
        model.load_state_dict(state_dict)
    except:
        model = torch.load(file_path)

    model.to(device)

    cap = cv2.VideoCapture('../data/vid-20230819-224616_xrdC8N2x.mp4')
    while cap.isOpened():
        ret, frame = cap.read()

        processed = process_image(frame, args.image_size)
        mask = predict(model, processed, device)

        colormap = create_colormap()
        frame = cv2.resize(frame, (args.image_size, args.image_size))

        image_combined = visualize_semantic(mask, colormap, frame, 0.3)
        cv2.imshow('frame', cv2.resize(image_combined, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def predict(model, image, device):
    x = image.to(device)
    output = model(x.float().unsqueeze(0)).cpu()
    output = F.interpolate(output, (x.shape[1], x.shape[2]), mode='bilinear')

    mask = output.argmax(dim=1)
    mask = mask[0].long().squeeze().numpy()

    coords = np.argwhere(np.isin(mask, DESIRED_LABELS)).T
    ground = np.zeros((mask.shape[0], mask.shape[1]))
    ground[coords[0], coords[1]] = 1

    return ground


def process_image(img, size):
    img = cv2.resize(img, (size, size))

    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy())
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("-s", "--image-size", type=int, default=256, help="target input image size (default: 256)")
    parser.add_argument("-n", "--file-name", help="model file name")
    args = parser.parse_args()
    main(args)
