import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor

class DiceLoss(nn.LossBase):
    def __init__(self, smooth=1e-7, reduction="mean"):
        super(DiceLoss, self).__init__(reduction)
        self.reshape = P.Reshape()
        self.smooth = smooth

    def construct(self, logits, label):
        intersection = self.reduce_sum(self.mul(logits.view(-1), label.view(-1)))
        unionset = self.reduce_sum(self.mul(logits.view(-1), logits.view(-1))) + \
                   self.reduce_sum(self.mul(label.view(-1), label.view(-1)))

        single_dice_coeff = (2 * intersection) / (unionset + self.smooth)
        dice_loss = 1 - single_dice_coeff / label.shape[0]
        return dice_loss


class MultiClassDiceLoss(nn.LossBase):
    def __init__(self, weights=None, ignore_indiex=None):
        super(MultiClassDiceLoss, self).__init__()
        self.binarydiceloss = DiceLoss()

    def construct(self, logits, label, weight=None):

        total_loss = 0
        for i in range(label.shape[1]):
            dice_loss = self.binarydiceloss(logits[:, i], label[:, i])
            if weight is not None:
                dice_loss *= weight[i]
            total_loss += dice_loss

        return total_loss
