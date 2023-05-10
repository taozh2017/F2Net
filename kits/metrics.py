import torch
import numpy as np
from hausdorff import hausdorff_distance


class SegMetrics(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def PA(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def mPA(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def dice(self):
        TN = self.confusion_matrix[0][0]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[1][1]
        FP = self.confusion_matrix[0][1]
        dice = 2 * TP / (2 * TP + FP + FN)
        return dice

    def sensitivity(self):
        TN = self.confusion_matrix[0][0]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[1][1]
        FP = self.confusion_matrix[0][1]
        dice = TP / (TP + FN)
        return dice

    def specificity(self):
        TN = self.confusion_matrix[0][0]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[1][1]
        FP = self.confusion_matrix[0][1]
        dice = TN / (TN + FP)
        return dice

    def mIoU(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def IoU(self):
        TN = self.confusion_matrix[0][0]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[1][1]
        FP = self.confusion_matrix[0][1]
        IoU = TP / (TP + FP + FN)
        return IoU

    def FWIoU(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def update(self, pre_image, gt_image):
        # assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def dice_coeff(pred, target):
    smooth = 1e-7

    intersection = (pred * target).sum()

    return (2. * intersection + smooth) / \
           (pred.sum() + target.sum() + smooth)
    '''
    smooth = 1e-7
    num = target.size(0)
    input_flat = pred.view(num, -1)  # Flatten
    target_flat = target.view(num, -1)  # Flatten
    intersection = input_flat * target_flat

    dice = 2 * (intersection.sum(1)) / (input_flat.sum(1) + target_flat.sum(1) + smooth)

    return dice.sum() / num
    '''


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def hausdorff_95(pred, target):
    num = target.shape[0]
    h = 0
    for i in range(num):
        h += hausdorff_distance(pred[i], target[i], distance="euclidean")
    return (h / num) * 0.95
