from avgmentations import augmentations as A, utils
from avgmentations.resnet_dataset import RESNET_MEAN, RESNET_STD
from torchvision import transforms
from torchvision.datasets import ImageFolder

class Experiment():

    init_transform = [
        transforms.ToTensor(),
        transforms.Normalize(RESNET_MEAN, RESNET_STD),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]

    def __init__(self, root, n_classes=1000):
        self.default_dataset = ImageFolder(
            root = root,
            transform = transforms.Compose(Experiment.init_transform),
            target_transform = A.OneHot(n_classes)
        )
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def default(self):
        raise NotImplementedError

    def basic(self):
        raise NotImplementedError
    
    def mixup(self, alpha=0.2, min_lam=0.3, max_lam=0.7):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ])
        }

        return exp
    
    def early_mixup(self, stop_point, alpha=0.2, min_lam=0.3, max_lam=0.7):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ]),
            stop_point: transforms.Compose(Experiment.init_transform)
        }
        
        return exp
    
    def p_mixup(self, p, alpha=0.2, min_lam=0.3, max_lam=0.7):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                A.P(
                    A.MixUp(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam), 
                    p=p
                ),
            ])
        }

        return exp
    
    def cutmix(self, alpha=0.2, min_lam=0.3, max_lam=0.7):
        exp = {
            0: A.Compose(Experiment.init_transform + [
                A.CutMix(self.default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam),
            ])
        }

        return exp