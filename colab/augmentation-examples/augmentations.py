import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
from PIL import Image


class Double():

    def __init__(self, spread=50):
        self.spread = spread

    def __call__(self, img):
        img1 = img.transform(img.size, Image.AFFINE, (1, 0, self.spread / 2, 0, 1, 0))
        img2 = img.transform(img.size, Image.AFFINE, (1, 0, -self.spread / 2, 0, 1, 0))
        return Image.blend(img1, img2, 0.5)


class CutOut(object):

    """
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):

        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        height = img.size(1)
        width = img.size(2)
        
        mask = np.ones((height, width), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(height)
            x = np.random.randint(width)

            y1 = np.clip(y - self.length // 2, 0, height)
            y2 = np.clip(y + self.length // 2, 0, height)
            x1 = np.clip(x - self.length // 2, 0, width)
            x2 = np.clip(x + self.length // 2, 0, width)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Gridmask(object):

    def __init__(self):

        """
        Randomly mask out four patches from an image.
        Args:
            no arg
        """

        pass

    def __call__(self, img):

        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        height = img.size(1)
        width = img.size(2)
        
        mask = np.ones((height, width), np.float32)

        y = np.random.randint(height // 2)
        x = np.random.randint(width // 2)

        y_length = np.random.randint(y, height // 2)
        x_length = np.random.randint(x, width // 2)

        mask[y: y_length, x: x_length] = 0.
        mask[y + height // 2: y_length + height // 2, x: x_length] = 0.    
        mask[y: y_length, x + width // 2: x_length + width // 2] = 0.
        mask[y + height // 2: y_length + height // 2, x + width // 2: x_length + width // 2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)


def mixup(data, targets, alpha, n_classes):

    '''
    alpha: input parameter for beta distribution to calculate opacities of mixup (lambda)
    '''

    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets
