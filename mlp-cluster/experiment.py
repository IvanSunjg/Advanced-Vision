from avgmentations import augmentations as A, utils
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

    def basic(self):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
            ])
        }
        return exp
    
    def mixup(self, alpha=1.0, min_lam=0.3, max_lam=0.7):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ])
        }
        return exp
    
    def early_mixup(self, stop_point=5, alpha=1.0, min_lam=0.3, max_lam=0.7):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ]),
            stop_point: transforms.Compose(Experiment.init_transform)
        }
        return exp
    
    def p_mixup(self, p, alpha=1.0, min_lam=0.3, max_lam=0.7):
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
    
    def cutmix(self, alpha=1.0, min_lam=0.3, max_lam=0.7):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                A.CutMix(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ])
        }
        return exp
