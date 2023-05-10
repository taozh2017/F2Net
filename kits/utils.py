import torch
import torch.nn as nn


__all__ = [
    'generate_params'
]


def generate_params(model, key):
    """define a generator
    will be convert to a param list by list(params) in optimizer.add_param_group() method
    """
    if key == 'conv_weight':
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                yield m.weight
    if key == 'conv_bias':
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)) and m.bias is not None:
                yield m.bias
    if key == 'bn_weight':
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                yield m.weight
    if key == 'bn_bias':
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) and m.bias is not None:
                yield m.bias
    # any type you want to specify

