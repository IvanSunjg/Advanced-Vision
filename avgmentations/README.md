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
from avgmentations.resnet_dataset import ResNetImageFolder, RESNET_MEAN, RESNET_STD
from avgmentations import augmentations as A

# Create dataset for image mixing for MixUp and CutMix
default_dataset = ImageFolder(
    root = 'train',
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(RESNET_MEAN, RESNET_STD),
        transforms.Resize(256),
        transforms.RandomCrop(224)
    ]),
    target_transform = A.OneHot(1000)
)

# Create dataset that applies MixUp, then CutMix 
# with a probability of 0.5, and finally GridMask
# to each image
dataset = ResNetImageFolder(
    'train',
    {
        0: A.Compose([
            transforms.ToTensor(),
            transforms.Normalize(RESNET_MEAN, RESNET_STD),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),

            A.MixUp(default_dataset),
            A.P(
                A.CutMix(default_dataset),
                p=0.5
            ),
            A.GridMask()
        ])
    }
)
```

## Notes

* **AV**gmentations implements a version of `Compose` that allows transformations to transform both the image and label simultaneously. This functionality is required for `avgmentations.MixUp` and `avgmentations.CutMix`, so if you use those in addition to other transforms, use `avgmentations.Compose` instead of `transforms.Compose`.
* Augmentations that require mixing of different images such as `MixUp` and `CutMix` require a second dataset to be defined to load in images to be mixed with. In the example above, an `ImageFolder` is used as this mixing dataset and is passed to the `MixUp` and `CutMix` augmentations.

## Making Changes

If you want to edit or add new augmentations or functionalities to this library, make the edit and then update the version number in [`setup.py`](setup.py#L8).
After the changes have been made and pushed to this repository, you can upgrade your version of `avgmentations` by running the installation command again:

For Linux and MacOS:

```sh
pip install 'git+https://github.com/IvanSunjg/Advanced-Vision.git#egg=avgmentations&subdirectory=avgmentations'
```
