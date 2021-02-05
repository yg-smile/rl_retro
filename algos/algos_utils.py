from skimage.color import rgb2gray
import numpy as np
import torch


def phi(img):
    # input: img: uint8 numpy array with 3 color channels with shape (h, w, 3)
    # output: tensor with 1 channel with shape (1, h, w)
    gray_scale = np.uint8(255 * rgb2gray(img))
    return torch.from_numpy(gray_scale)[None, :, :]
