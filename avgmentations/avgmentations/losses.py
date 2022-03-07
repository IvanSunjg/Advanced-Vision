import utils

class CrossEntropyLoss():
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, data, target):
        return utils.cross_entropy_loss(data, target, self.size_average)

class BCEEntropyLoss():
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, data, target):
        return utils.bceloss(data, target)