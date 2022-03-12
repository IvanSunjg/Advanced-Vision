from avgmentations import augmentations as A
from avgmentations.resnet_dataset import RESNET_MEAN, RESNET_STD
from torchvision import transforms
from torchvision.datasets import ImageFolder

class Experiment():

    init_transform = [
        transforms.ToTensor(),
        transforms.Normalize(RESNET_MEAN, RESNET_STD),
    ]

    def __init__(self, root, n_classes=1000):
        self.default_dataset = ImageFolder(
            root = root,
            transform = transforms.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]),
            target_transform = A.OneHot(n_classes)
        )
    
    def construct_experiment(self, exp_type, log=True, **exp_kwargs):
        if log:
            print(f'{exp_type} {exp_kwargs}')
        return getattr(self, exp_type)(**exp_kwargs)

    def basic(self, sharpness_factor=2):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor),
            ])
        }
        return exp
    
    def augmix(self, k=3, w1=0.2, w2=0.3, w3=0.5, m=0.2):
        exp = {
            0: transforms.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.AugMix(k=k, w1=w1, w2=w2, w3=w3, m=m),
            ])
        }
        return exp
    
    def mixup(self, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ])
        }
        return exp
    
    def mixup_then_default(self, stop_point=5, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ]),
            stop_point: transforms.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
            ])
        }
        return exp
    
    def p_mixup(self, p, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.P(
                    A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam), 
                    p=p
                ),
            ])
        }
        return exp
    
    def mixup_then_basic(self, stop_point=5, alpha=1.0, min_lam=0, max_lam=1, sharpness_factor=2):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ]),
            stop_point: transforms.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor),
            ])
        }
        return exp
    
    def cutmix(self, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.CutMix(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ])
        }
        return exp
    
    def cutmix_then_default(self, stop_point=5, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.CutMix(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ]),
            stop_point: transforms.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
            ])
        }
        return exp
    
    def p_cutmix(self, p, alpha=1.0, min_lam=0, max_lam=1):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.P(
                    A.CutMix(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam), 
                    p=p
                ),
            ])
        }
        return exp
