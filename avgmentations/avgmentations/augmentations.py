import numpy as np
from torchvision import transforms
import utils

class OneHot():
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def __call__(self, target):
        return utils.one_hot(target, self.n_classes)

class ItemTransform():

    '''
    An abstract class that is used to indicate when a transform is to be transformed on an item,
    i.e. an image-target pair. Used by transforms including, but not limited to, MixUp and CutMix.
    '''

    def __call__(self, img, label):
        raise NotImplementedError

def apply_transform(t, img, label=None):

    if isinstance(t, ItemTransform):
        img, label = t(img, label)
    else:
        img = t(img)
    
    return img, label

class Compose(ItemTransform, transforms.Compose):

    '''
    A version of `transforms.Compose` that allows for the optional transformation of items, i.e.
    image-target pairs.
    '''

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = apply_transform(t, img, label)
            
        return img, label

class Double():

    def __init__(self, spread=50):
        self.spread = spread

    def __call__(self, img):
        img = utils.double(img, self.spread)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(spread={self.spread})'

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

        img = utils.cutout(img, self.n_holes, self.length)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(n_holes={self.n_holes}, length={self.length})'

class GridMask():

    def __init__(self):

        """
        Randomly mask out four patches from an image.
        Args:
            no arg
        """

        pass

    def __call__(self, img):
        # TODO fix return description
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of size (length, length) cut out of it.
        """

        img = utils.gridmask(img)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class AugMix():
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

    def __init__(self, k=3, w1=0.2, w2=0.3, w3=0.5, m=0.2):
        self.k = k
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.m = m

    def __call__(self, img):
        '''
        Args:
            img (Tensor): Tensor image of size (C, H, W)
        '''

        # TODO could modify different augmentation method hyperparameters
        miximg = utils.augmix(img, k=self.k, w1=self.w1, w2=self.w2, w3=self.w3, m=self.m)
        return miximg
    
    def __repr__(self):
        return self.__class__.__name__ + f'(k={self.k}, w1={self.w1}, w2={self.w2}, w3={self.w3}, m={self.m})'

class OneOf(ItemTransform):

    def __init__(self, transforms, p=None):
        if p is not None:
            if len(transforms) != len(p):
                raise ValueError('Transforms and probabilities have to have the same length when probabilities are specified.')
            if sum(p) != 1:
                raise ValueError('Sum of probabilities must equal 1.')

        self.transforms = transforms
        self.p = p

    def __call__(self, img, label):
        t = np.random.choice(self.transforms, p=self.p)
        img, label = apply_transform(t, img, label)
        return img, label
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        tp = zip(self.transforms, self.p) if self.p is not None else self.transforms
        for t in tp:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class P(ItemTransform):

    '''
    Apply a transformation with a probability.
    '''

    def __init__(self, transform, p):
        self.transform = transform
        self.p = p
    
    def __call__(self, img, label):
        if np.random.rand() < self.p:
            return apply_transform(self.transform, img, label)
        return img, label
    
    def __repr__(self):
        return self.__class__.__name__ + f'({self.transform}, {self.p})'

class Mix():

    '''
    Superclass for transformations that require image mixing.
    '''

    def __init__(self, dataset):
        self.dataset = dataset
    
    def get_mix_item(self):
        mix_idx = np.random.randint(len(self.dataset))
        mix_image, mix_target = self.dataset[mix_idx]

        return mix_image, mix_target

class MixUp(Mix, ItemTransform):

    def __init__(self, dataset, alpha=0.2, min_lam=0.3, max_lam=0.7):
        super().__init__(dataset)

        self.alpha = alpha
        self.min_lam = min_lam
        self.max_lam = max_lam

    def __call__(self, img, label):
        mix_image, mix_label = super().get_mix_item()
        img, label = utils.mixup(img, label, mix_image, mix_label, alpha=self.alpha, min_lam=self.min_lam, max_lam=self.max_lam)

        return img, label
    
    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, min_lam={self.min_lam}, max_lam={self.max_lam})'

class CutMix(Mix, ItemTransform):

    def __init__(self, dataset, alpha=0.2, min_lam=0.3, max_lam=0.7):
        super().__init__(dataset)

        self.alpha = alpha
        self.min_lam = min_lam
        self.max_lam = max_lam

    def __call__(self, img, label):
        mix_image, mix_label = super().get_mix_item()
        img, label = utils.cutmix(img, label, mix_image, mix_label, alpha=self.alpha, min_lam=self.min_lam, max_lam=self.max_lam)

        return img, label
    
    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, min_lam={self.min_lam}, max_lam={self.max_lam})'
