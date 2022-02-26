import os
from turtle import width
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
from PIL import Image

class CustomImageFolder(ImageFolder):

    def __init__(self, root, epoch_transforms={}, *args, **kwargs):
        self.epoch_transforms = epoch_transforms
        super().__init__(root, *args, **kwargs)

    def update_transform(self, epoch):
        if epoch in self.epoch_transforms:
            self.transform = self.epoch_transforms[epoch]
            
