from avgmentations import augmentations as A
from avgmentations.resnet_dataset import RESNET_MEAN, RESNET_STD
from torchvision import transforms
from torchvision.datasets import ImageFolder

class Experiment():

    def __init__(self, root, n_classes=1000):
        self.default_dataset = ImageFolder(
            root = root,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.CenterCrop(224) # TODO Random crop?
            ]),
            target_transform = A.OneHot(n_classes)
        )
    
    def construct_experiment(self, exp_type, log=True, **exp_kwargs):
        if log:
            print(f'{exp_type} {exp_kwargs}')
        return getattr(self, exp_type)(**exp_kwargs)

    def default(self):
        exp = {
            0: transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224)
            ])
        }
        return exp

    def basic(self, sharpness_factor=2):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor)
            ])
        }
        return exp
    
    def cutout(self, n_holes=3, length=50):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.CutOut(n_holes=n_holes, length=length)
            ])
        }
        return exp
    
    def gridmask(self):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.GridMask()
            ])
        }
        return exp
    
    def augmix(self, k=3, w=[0.2, 0.3, 0.5], m=0.2, level=3):
        exp = {
            0: transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.AugMix(k=k, w=w, m=m, level=level),
            ])
        }
        return exp
    
    def mixup(self, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam)
            ])
        }
        return exp
    
    def mixup_then_default(self, stop_point=5, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam)
            ]),
            stop_point: transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224)
            ])
        }
        return exp
    
    def p_mixup(self, p, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.P(
                    A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam), 
                    p=p
                )
            ])
        }
        return exp
    
    def mixup_then_basic(self, stop_point=5, alpha=1.0, min_lam=0, max_lam=1, sharpness_factor=2):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam)
            ]),
            stop_point: transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor)
            ])
        }
        return exp
    
    def cutmix(self, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.CutMix(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam)
            ])
        }
        return exp
    
    def cutmix_then_default(self, stop_point=5, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.CutMix(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam)
            ]),
            stop_point: transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224)
            ])
        }
        return exp
    
    def p_cutmix(self, p, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.P(
                    A.CutMix(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam), 
                    p=p
                )
            ])
        }
        return exp
    
    def augmix_cutout(self, k=3, w=[0.2, 0.3, 0.5], m=0.2, level=3, n_holes=3, length=50):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.AugMix(k=k, w=w, m=m,level=level),
                A.CutOut(n_holes=n_holes, length=length)
            ])
        }
        return exp
    
    def basic_mixup(self, sharpness_factor=2, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor),

                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam)
            ])
        }
        return exp
    
    def mixup_then_basic_gridmask_then_default(self, n1, n2, alpha=1.0, min_lam=0, max_lam=1, sharpness_factor=2):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam)
            ]),
            n1: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor),

                A.GridMask()
            ]),
            n2: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224)
            ])
        }
        return exp
    
    def oneof__mixup_cutmix(self, p=[0.5, 0.5], alpha1=1, alpha2=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.OneOf(
                    transforms=[
                        A.MixUp(self.default_dataset, alpha=alpha1),
                        A.CutMix(self.default_dataset, alpha=alpha2)
                    ],
                    p=p
                )
            ])
        }
        return exp
    
    def mixup_mixup(self, alpha1=1, alpha2=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.MixUp(self.default_dataset, alpha=alpha1),
                A.MixUp(self.default_dataset, alpha=alpha2)
            ])
        }
        return exp
    
    def mixup_basic(self, alpha=1, sharpness_factor=2):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.MixUp(self.default_dataset, alpha=alpha),
                
                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor)
            ])
        }
        return exp
    
    def autoaugment(self):
        exp = {
            0: A.Compose([
                transforms.AutoAugment(),

                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),
            ])
        }
        return exp
    
    def autoaugment_mixup(self, alpha=1):
        exp = {
            0: A.Compose([
                transforms.AutoAugment(),

                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.MixUp(dataset=self.default_dataset, alpha=alpha)
            ])
        }
        return exp
    
    def cutout_mixup(self, n_holes=3, length=50, alpha=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.CutOut(n_holes=n_holes, length=length),
                A.MixUp(dataset=self.default_dataset, alpha=alpha)
            ])
        }
        return exp
    
    def basic_mixup_then_autoaugment(self, n=5, sharpness_factor=2, alpha=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor),
                A.MixUp(self.default_dataset, alpha=alpha)
            ]),
            n: A.Compose([
                transforms.AutoAugment(),

                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224)
            ])
        }
        return exp

    def basic_mixup_then_autoaugment_mixup(self, n=10, sharpness_factor=2, alpha1=1, alpha2=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor),
                A.MixUp(self.default_dataset, alpha=alpha1)
            ]),
            n: A.Compose([
                transforms.AutoAugment(),

                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.MixUp(self.default_dataset, alpha=alpha2)
            ])
        }
        return exp

    def basic_then_autoaugment(self, n=5, sharpness_factor=2):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor)
            ]),
            n: A.Compose([
                transforms.AutoAugment(),

                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224)
            ])
        }
        return exp
    
    def basic_then_mixup(self, n=5, sharpness_factor=2, alpha=1):
        exp = {
            0: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor)
            ]),
            n: A.Compose([
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.RandomCrop(224),

                A.MixUp(self.default_dataset, alpha=alpha)
            ])
        }
        return exp
