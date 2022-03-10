from torchvision.datasets import ImageFolder
from . import augmentations
import numpy as np

# Mean and Standard Deviation of full ImageNet dataset
RESNET_MEAN = np.array([0.485, 0.456, 0.406])
RESNET_STD = np.array([0.229, 0.224, 0.225])

class ResNetImageFolder(ImageFolder):

    def __init__(self, root, epoch_transforms={}, **kwargs):
        super().__init__(root, **kwargs)

        self.epoch_transforms = epoch_transforms
        self.n_classes = len(self.classes)
        self.ohe_transform = augmentations.OneHot(self.n_classes)

        if 0 in self.epoch_transforms:
            self.transform = self.epoch_transforms[0]

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        target = self.ohe_transform(target)

        if self.transform is not None:
            sample, target = augmentations.apply_transform(self.transform, sample, target)
        return sample, target

    def update_transform(self, epoch):
        if epoch in self.epoch_transforms:
            self.transform = self.epoch_transforms[epoch]
