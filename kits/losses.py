import torch
import torch.nn.functional as f


class BinaryCrossEntropy(torch.nn.Module):
    """ Binary CrossEntropy Wrapper for mutiple logits input
    force to activate the output tensor with sigmoid
    """

    def __init__(self, reduction="mean"):
        super(BinaryCrossEntropy, self).__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def forward(self, output, target, weight=None):
        # convert to a list of logits
        # if not isinstance(output, (list, tuple)):
        #    output = [output]

        # generate a default weight
        # if weight is None:
        #    weight = [1. / len(output)] * len(output)

        loss = 0
        # target = torch.unsqueeze(target, 1)
        n, c, h, w = output.size()
        criterion = torch.nn.BCELoss(weight=weight, reduction='mean').cuda()
        loss = criterion(output, target.float())
        loss /= n
        '''
        for o, w in zip(output, weight):
            if o.size()[2:] != target.size()[2:]:
                t = torch.nn.functional.interpolate(target, o.size()[2:], mode='nearest')
            else:
                t = target
            loss += f.binary_cross_entropy_with_logits(o, t, reduction=self.reduction) * w
        '''
        return loss


class CategoricalCrossEntropy(torch.nn.Module):
    """ Categorical CrossEntropy Wrapper for mutiple logits input
    integrate a optional activate layer for ouput, the activation is limited to softmax(default) or sigmoid(for multiple label)
    """

    def __init__(self, reduction="mean"):
        super(CategoricalCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, output, target, weight=None):
        # convert to a list of logits
        if not isinstance(output, (list, tuple)):
            output = [output]

        # generate a default weight
        if weight is None:
            weight = [1. / len(output)] * len(output)

        loss = 0

        for o, w in zip(output, weight):
            if o.size()[2:] != target.size()[1:]:
                t = torch.nn.functional.interpolate(target, o.size()[2:], mode='nearest')
            else:
                t = target
            loss += f.cross_entropy(o, t.long(), reduction=self.reduction) * w

        return loss


def binary_focal_loss(logits, target, alpha=0.5, gamma=2, reduction="mean"):
    pt = torch.sigmoid(logits)
    ones = torch.ones_like(pt)
    loss0 = -alpha * (ones - pt) ** gamma * target * torch.log(pt)
    loss1 = (1 - alpha) * pt ** gamma * (ones - target) * torch.log(ones + torch.exp(logits))
    loss = loss0 + loss1

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    return loss


class BinaryFocalLoss(torch.nn.Module):
    r"""This criterion combines `cross_entropy` and `focal` in a single function.
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
                in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
                in the case of K-dimensional loss.
        target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
                or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
                K-dimensional loss.
        alpha:
        gamma:
        reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        """

    def __init__(self, alpha=0.5, gamma=2, reduction="mean"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target, weight=None):
        # convert to a list of logits
        if not isinstance(output, (list, tuple)):
            output = [output]

        # generate a default weight
        if weight is None:
            weight = [1. / len(output)] * len(output)

        loss = 0

        for o, w in zip(output, weight):
            if o.size()[2:] != target.size()[2:]:
                t = torch.nn.functional.interpolate(target, o.size()[2:], mode='nearest')
            else:
                t = target
            loss += binary_focal_loss(o, t, self.alpha, self.gamma, reduction=self.reduction) * w

        return loss


def categorical_focal_loss(input, target, gamma=2, reduction="mean"):
    # type: (torch.Tensor, torch.LongTensor, float, str) -> torch.Tensor
    """
    :param input: a tensor input contains at least [B, C], or [B, C, K_0, K_1, ...]
    :param target: a categorical-index tensor correspond to ```input''', size should be [B, K_0, K_1, ...]
    :param gamma: focus factor
    :param reduction: reduction mode
    :return: a loss tensor
    """

    n, c = input.size(0), input.size(1)
    # the input and target should have same dimension
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format((n,) + input.size()[2:], target.size()))

    ce = f.nll_loss(f.log_softmax(input, dim=1), target, reduction='none')
    focus = torch.sub(1., torch.exp(-ce)) ** gamma

    loss = focus * ce

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)

    return loss


class CategoricalFocalLoss(torch.nn.Module):
    r"""This criterion combines `cross_entropy` and `focal` in a single function.
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
                in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
                in the case of K-dimensional loss.
        target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
                or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
                K-dimensional loss.
        alpha:
        gamma:
        reduction (string, optional): Specifies the reduction to apply to the output:
    """

    def __init__(self, gamma=2, reduction="mean"):
        super(CategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target, weight=None):
        # convert to a list of logits
        if not isinstance(output, (list, tuple)):
            output = [output]

        # generate a default weight
        if weight is None:
            weight = [1. / len(output)] * len(output)

        loss = 0

        for o, w in zip(output, weight):

            if o.size()[2:] != target.size()[2:]:
                t = torch.nn.functional.interpolate(target, o.size()[2:], mode='nearest')
            else:
                t = target
            loss += categorical_focal_loss(o, t.long(), self.gamma, reduction=self.reduction) * w

        return loss


class BinaryDiceLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1e-7

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat

        dice = 2 * (intersection.sum(1)) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # print(dice.sum()/N)
        loss = 1 - dice.sum() / N

        return loss


class MulticlassDiceLoss(torch.nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = BinaryDiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
            
        return totalLoss


class WiouWbceLoss(torch.nn.Module):
    def __init__(self):
        super(WiouWbceLoss, self).__init__()

    def forward(self, input, target):
        weit = 1 + 5 * torch.abs(f.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = f.binary_cross_entropy_with_logits(input, target, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        input = torch.sigmoid(input)
        inter = ((input * target) * weit).sum(dim=(2, 3))
        union = ((input + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

