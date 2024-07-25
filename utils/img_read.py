import numpy as np
from PIL import Image
import os
from kornia import image_to_tensor, tensor_to_image
from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb
import torch


def img_read(path, mode):
    '''
    input: path, mode
    output: tensor, [c, h, w]
    '''
    assert mode == 'RGB' or mode == 'L' or mode == 'YCbCr'  # RGB、灰度图、YCbCr

    if mode == 'RGB' or mode == 'L':
        img = np.asarray(Image.open(path).convert(mode), dtype=np.float32)
        img = image_to_tensor(img, keepdim=True) / 255.0
        return img
    elif mode == 'YCbCr':
        img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
        img = image_to_tensor(img, keepdim=True) / 255.0
        img = rgb_to_ycbcr(img)
        y, cbcr = torch.split(img, [1, 2], dim=0)
        return y, cbcr


def img_save(image, imagename, savedir, mode='L'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    img = Image.fromarray(image, mode=mode)
    path = os.path.join(savedir, imagename)
    img.save(path)
