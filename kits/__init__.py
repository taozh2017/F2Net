from .metrics import *


from .schedulers import LR_Scheduler

from .losses import BinaryCrossEntropy
from .losses import CategoricalCrossEntropy
from .losses import BinaryFocalLoss
from .losses import CategoricalFocalLoss
from .losses import MulticlassDiceLoss
from .losses import WiouWbceLoss

from .utils import generate_params

from .logger import setup_logger

from .saver import Saver
from .summaries import TensorboardSummary


loss_names = ('bce', 'ce', 'bfocal', 'focal', 'dice', 'wiouwbce')

def configure_loss(loss_name, **kwargs):
    _loss = None
    if loss_name == 'bce':
        _loss = BinaryCrossEntropy(**kwargs)
    elif loss_name == 'ce':
        _loss = CategoricalCrossEntropy(**kwargs)
    elif loss_name == 'bfocal':
        _loss = BinaryFocalLoss(**kwargs)
    elif loss_name == 'focal':
        _loss = CategoricalFocalLoss(**kwargs)
    elif loss_name == 'dice':
        _loss = MulticlassDiceLoss(**kwargs)
    elif loss_name == 'wiouwbce':
        _loss = WiouWbceLoss(**kwargs)
    else:
        raise NotImplementedError('loss {} is not recognized as a valid loss'.format(loss_names))

    return _loss
