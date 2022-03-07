# AVgmentations

A python library that provides data augmentations for the Image Recognition on Limited Data Challenge. It is named AVgmentations, as it provides data augmentations tools and was initially developed for the University of Edinburgh's Advanced Vision (AV) course, hence a play on words, **AV**gmentations.

AVgmentations implements a modified version of the pytorch data loading and augmentation pipeline that allows for transformation of both the sample and target simultaneously. This allows for more complex data augmentations such as MixUp and CutMix to be used modularly.

The implementations also allows for the use of transformations from the original pytorch library as well as the option to use different augmentation techniques at different training epochs.

## Dependencies

* pytorch-gpu (or cpu)
* torchvision
* numpy
* matplotlib

## Installation
The library can be installed from GitHub using `pip`.

For Linux and MacOS:

```bash
pip install 'git+https://github.com/IvanSunjg/Advanced-Vision.git#egg=avgmentations&subdirectory=avgmentations'
```

For Windows:

```bash
pip install 'git+https://github.com/IvanSunjg/Advanced-Vision.git#egg=avgmentations^&subdirectory=avgmentations'
```

## Getting Started
After installing this library, it can be used directly in python scripts. The custom written dataset can be found under `avgmentations.resnet_dataset` and the custom augmentations under `avgmentations.augmentations`.

Example use for loading a dataset that uses multiple complex data augmentation techniques:

```python
from torchvision.datasets import ImageFolder
from avgmentations.resnet_dataset import ResNetImageFolder
from avgmentations import augmentations as augs

default_dataset = ImageFolder(
    root = 'train',
    transform = augs.DEFAULT_AUGMENTATION,
    target_transform = augs.OneHot(1000)
)

dataset = ResNetImageFolder(
    'train',
    {
        0: augs.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),

            augs.MixUp(default_dataset),
            augs.P(augs.CutMix(default_dataset), p=0.5),
            augs.GridMask()
        ])
    }
)
```
