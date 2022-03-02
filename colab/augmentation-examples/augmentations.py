import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import random


class Double():

    def __init__(self, spread=50):
        self.spread = spread

    def __call__(self, img):
        img1 = img.transform(img.size, Image.AFFINE, (1, 0, self.spread / 2, 0, 1, 0))
        img2 = img.transform(img.size, Image.AFFINE, (1, 0, -self.spread / 2, 0, 1, 0))
        return Image.blend(img1, img2, 0.5)


class CutOut():

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

class Gridmask():

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

class CornerMask():

    """
    Mask four corners with patches from an image.
    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self,length):

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

        mask[: self.length, : self.length] = 0.
        mask[-self.length: , : self.length] = 0.
        mask[: self.length, -self.length: ] = 0.
        mask[-self.length: , -self.length: ] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

## for checking the output
def augmix(img,k=3, w1=0.2, w2=0.3, w3=0.5, m=0.2):
    '''
    @article{hendrycks2020augmix,
    title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
    author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2020}
    }

    k: number of different augumentations taken (default 3)
    w1,w2,w3: weight for each augumentated image to mixup
    m: weight for mix with the original and the mixup augumentated image
    '''

    ## could modify different augmentation method' hyperparameters
    augulist = ["autocontrast", "rotate", "translate_x", "translate_y", "shear_x", "shear_y"]
    selects = random.sample(augulist, k)
    images = []

    for i in range(len(selects)):

        if selects[i] == "autocontrast":
            new_image = transforms.functional.autocontrast(img)
            images.append(new_image)


        elif selects[i] == "rotate":
            # small rotation degree in order to keep the image not to be destroyed 
            new_image = transforms.functional.rotate(img, random.randint(-10,10))
            images.append(new_image)
            
        elif selects[i] == "translate_x":
            new_image = transforms.functional.affine(img, translate=(random.uniform(-20,20),0),angle=0, scale = 1,shear = 0)
            images.append(new_image)

        elif selects[i] == "translate_y":
            new_image = transforms.functional.affine(img, translate=(0,random.uniform(-20,20)),angle=0, scale = 1,shear = 0)
            images.append(new_image)
               
        elif selects[i] == "shear_x":
            new_image = transforms.functional.affine(img, translate=(0,0),angle=0, scale = 1,shear = (random.uniform(-20,20),0))
            images.append(new_image)

        elif selects[i] == "shear_y":
            new_image = transforms.functional.affine(img, translate=(0,0),angle=0, scale = 1,shear = (0,random.uniform(-20,20)))
            images.append(new_image)

    mixed = torch.mul(images[0],w1) + torch.mul(images[1],w2) + torch.mul(images[2],w3)
    miximg = torch.mul(mixed,1-m) + torch.mul(img,m)
    
    
    return mixed, miximg, images[0],images[1],images[2]



class AugMix(object):
    '''
    @article{hendrycks2020augmix,
    title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
    author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2020}
    }

    k: number of different augumentations taken (default 3)
    w1,w2,w3: weight for each augumentated image to mixup
    m: weight for mix with the original and the mixup augumentated image
    '''

    def __init__(self,k=3, w1=0.2, w2=0.3, w3=0.5, m=0.2):
        self.k = k
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.m = m

    def __call__(self, img):

        """
        Args:
            img (Tensor): Tensor image of size (C, H, W)
        """

        ## could modify different augmentation method' hyperparameters
        augulist = ["autocontrast", "rotate", "translate_x", "translate_y", "shear_x", "shear_y"]
        selects = random.sample(augulist, self.k)
        images = []

        for i in range(len(selects)):

            if selects[i] == "autocontrast":
                new_image = transforms.functional.autocontrast(img)
                images.append(new_image)

            elif selects[i] == "rotate":
                # small rotation degree in order to keep the image not to be destroyed 
                new_image = transforms.functional.rotate(img, random.randint(-10,10))
                images.append(new_image)
            
            elif selects[i] == "translate_x":
                new_image = transforms.functional.affine(img, translate=(random.uniform(-20,20),0),angle=0, scale = 1,shear = 0)
                images.append(new_image)

            elif selects[i] == "translate_y":
                new_image = transforms.functional.affine(img, translate=(0,random.uniform(-20,20)),angle=0, scale = 1,shear = 0)
                images.append(new_image)
               
            elif selects[i] == "shear_x":
                new_image = transforms.functional.affine(img, translate=(0,0),angle=0, scale = 1,shear = (random.uniform(-20,20),0))
                images.append(new_image)

            elif selects[i] == "shear_y":
                new_image = transforms.functional.affine(img, translate=(0,0),angle=0, scale = 1,shear = (0,random.uniform(-20,20)))
                images.append(new_image)

        mixed = torch.mul(images[0],self.w1) + torch.mul(images[1],self.w2) + torch.mul(images[2],self.w3)
        miximg = torch.mul(mixed,1-self.m) + torch.mul(img,self.m)
        return miximg


def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(1, label.view(-1, 1), 1)


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


def cross_entropy_loss(data, target, size_average=True):
    data = F.log_softmax(data, dim =1)
    loss = -torch.sum(data * target)
    if size_average:
        return loss / data.size(0)
    else:
        return loss


class CrossEntropyLoss():
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, data, target):
        return cross_entropy_loss(data, target, self.size_average)
