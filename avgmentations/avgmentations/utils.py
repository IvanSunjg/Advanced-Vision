import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from .resnet_dataset import RESNET_MEAN, RESNET_STD

def imshow(inp, title='untitled', norm=True):
    """Imshow for Tensor Images."""
    
    inp = inp.numpy().transpose((1, 2, 0))
    if norm:
        inp = RESNET_STD * inp + RESNET_MEAN
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

def one_hot(target, n_classes):
    label = torch.zeros(n_classes)
    label[target] = 1.
    return label

def double(img, spread):
    img1 = transforms.functional.affine(img, translate=(spread, 0), angle=0, scale = 1, shear = 0)
    img2 = transforms.functional.affine(img, translate=(-spread, 0), angle=0, scale = 1, shear = 0)
    img = 0.5 * img1 + 0.5 * img2

    return img

def augmix(img, k=3, w = [0.2, 0.3, 0.5], m=0.2, level = 3):
    '''
    @article{hendrycks2020augmix,
    title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
    author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2020}
    }

    k: number of different augmentations taken (default 3)
    w1,w2,w3: weight for each augmentated image to mixup
    m: weight for mix with the original and the mixup augmentated image
    level: level of augmention
    '''

    auglist = ["hflip", "vflip", "autocontrast", "rotate", "translate_x", "translate_y", "shear_x", "shear_y"]
    augs = np.random.choice(auglist, k)
    images = []
    for aug in augs:
        if aug == "hflip":
            new_image = transforms.functional.hflip(img)
        elif aug == "vflip":
            new_image = transforms.functional.vflip(img)
        elif aug == "autocontrast":
            new_image = transforms.functional.autocontrast(img)
        elif aug == "rotate":
            # small rotation degree in order to keep the image not to be destroyed 
            new_image = transforms.functional.rotate(img, np.random.randint(-10 * level, 10 * level))
        elif aug == "translate_x":
            new_image = transforms.functional.affine(img, translate=(np.random.uniform(-10 * level, 10 * level), 0), angle=0, scale=1, shear=0)
        elif aug == "translate_y":
            new_image = transforms.functional.affine(img, translate=(0, np.random.uniform(-10 * level, 10 * level)), angle=0, scale=1, shear=0)
        elif aug == "shear_x":
            new_image = transforms.functional.affine(img, translate=(0,0),angle=0, scale = 1,shear = (np.random.uniform(-10 * level, 10 * level),0))
        elif aug == "shear_y":
            new_image = transforms.functional.affine(img, translate=(0,0),angle=0, scale = 1,shear = (0,np.absrandom.uniform(-10 * level, 10 * level)))

        images.append(new_image)

    mixed = torch.zeros_like(img)
    for i in range(k):
        mixed += torch.mul(images[i], w[i])

    miximg = torch.mul(mixed, 1 - m) + torch.mul(img, m)
    
    return miximg

def cutout(img, n_holes, length):
    height = img.size(1)
    width = img.size(2)
    
    mask = np.ones((height, width), np.float32)

    for _ in range(n_holes):
        y = np.random.randint(height)
        x = np.random.randint(width)

        y1 = np.clip(y - length // 2, 0, height)
        y2 = np.clip(y + length // 2, 0, height)
        x1 = np.clip(x - length // 2, 0, width)
        x2 = np.clip(x + length // 2, 0, width)

        mask[y1:y2, x1:x2] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask

    return img

def gridmask(img):
    height = img.size(1)
    width = img.size(2)
    
    mask = np.ones((height, width), np.float32)

    y = np.random.randint(height // 2)
    x = np.random.randint(width // 2)

    y_length = np.random.randint(y, height // 2)
    x_length = np.random.randint(x, width // 2)

    mask[y:y_length, x:x_length] = 0.
    mask[y + height // 2:y_length + height // 2, x:x_length] = 0.    
    mask[y:y_length, x + width // 2:x_length + width // 2] = 0.
    mask[y + height // 2:y_length + height // 2, x + width // 2:x_length + width // 2] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask

    return img

def mixup(image1, label1, image2, label2, alpha=0.2, min_lam=0.3, max_lam=0.7):
    # Select a random number from the given beta distribution
    # Mixup the images accordingly
    alpha = 0.2
    lam = np.clip(np.random.beta(alpha, alpha), min_lam, max_lam)
    mixup_image = lam * image1 + (1 - lam) * image2
    mixup_label = lam * label1 + (1 - lam) * label2

    return mixup_image, mixup_label

def rand_bbox(h, w, lam):
    """
    Generate random bounding box 

    Args:
        - h: height of the bounding box
        - w: width of the bounding box
        - lam: (lambda) cut ratio parameter

    Returns:
        - Bounding box as 4-tuple
    """
    lam = 1. - lam
    cut_w = int(np.sqrt(lam * w * h))
    cut_h = int(np.sqrt(lam * w * h))

    # TODO discuss keeping this box behavior bc the box spills over the edge of the image
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2

def cutmix(image1, label1, image2, label2, alpha=0.2, min_lam=0, max_lam=1):
    lam = np.clip(np.random.beta(alpha, alpha), min_lam, max_lam)
    h, w = image1.shape[1:]
    bbx1, bby1, bbx2, bby2 = rand_bbox(h, w, lam)

    cutmix_image = image1
    cutmix_image[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (w * h)
    cutmix_label = label1 * lam + label2 * (1. - lam)

    return cutmix_image, cutmix_label

def cross_entropy_loss(data, target, size_average=True):
        data = F.log_softmax(data, dim=1)
        loss = -torch.sum(data * target, dim=1)
        if size_average:
            return loss.mean()
        else:
            return loss.sum()

def bceloss(x, y):
    eps = 1e-6
    loss = -torch.mean(y * torch.log(x + eps) + (1 - y) * torch.log(1 - x + eps))
    return loss