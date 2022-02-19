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


class CutOut():

    def __init__(self, size=(50, 50)):
        self.size = size
    
    def __call__(self, img):
        # cutout square
        np.random()
        pass


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
