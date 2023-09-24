import mindspore.nn as nn
from mindspore.ops import operations as P


class BinaryCrossEntropy(nn.LossBase):
    def __init__(self, weight=None, reduction='mean'):
        super(BinaryCrossEntropy, self).__init__()
        self.binary_cross_entropy = P.BinaryCrossEntropy(reduction=reduction)
        self.weight_one = weight is None
        if not self.weight_one:
            self.weight = weight
        else:
            self.ones = P.OnesLike()

    def construct(self, logits, labels, weight=None):
        if self.weight_one:
            weight = self.ones(logits)
        else:
            weight = self.weight

        loss = self.binary_cross_entropy(logits, labels, weight)

        return loss
