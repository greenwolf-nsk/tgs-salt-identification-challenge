__all__ = [
    'architectures',
    'losses',
    'optimizers',
    'schedulers',
]

from functools import partial

from torch.nn import BCEWithLogitsLoss

from .resnet_unet import UNetResNet34
from .wide_unet import TernausNetV2, WideUnet, WideUnetSE
from .seresnext_new import SeResNextUnet
from .deeply_supervised import DeeplySupervised34
from .losses import BCEJaccard, BCEDice, FocalLoss2d, bce_lovasz
from .combined_loss import CombinedLoss, CombinedLossHypercolumn, BoundaryLoss, BoundaryLoss2
from .lovasz_loss import lovasz_hinge
from .optimizers import adam_1e3, adam_1e4, adam_1e5, sgd_1e2, sgd_1e4, sgd_1e3
from .schedulers import step_scheduler_50_01, step_scheduler_50_05, step_scheduler_20_05, step_scheduler_100_07, cosine_annealing

architectures = {
    'unet_resnet34_v1': UNetResNet34,
    'wide_resnet_v1': TernausNetV2,
    'wide_resnet_v2': partial(TernausNetV2, num_filters=16),
    'wide_resnet_hypercolumn': WideUnet,
    'wide_resnet_se': WideUnetSE,
    'seresnext': SeResNextUnet,
    'ds_resnet_34': DeeplySupervised34,
    'wide_resnet_border': partial(TernausNetV2, num_classes=2),
}

losses = {
    'lovasz': lovasz_hinge,
    'focal': FocalLoss2d(),
    'BCE': BCEWithLogitsLoss(),
    'bce_lovasz': partial(bce_lovasz, alpha=0.1),
    'BCEJaccard': BCEJaccard(),
    'BCEDice': BCEDice(),
    'boundary_bce': BoundaryLoss(),
    'boundary_lovasz': BoundaryLoss2(weights=(0.5, 0.5)),
    'combined': CombinedLoss(lovasz_hinge),
    'combined_hyper': CombinedLossHypercolumn(lovasz_hinge),
}

optimizers = {
    'adam_1e-3': adam_1e3,
    'adam_1e-4': adam_1e4,
    'adam_1e-5': adam_1e5,
    'sgd_1e-2': sgd_1e2,
    'sgd_1e-3': sgd_1e3,
    'sgd_1e-4': sgd_1e4,
}

schedulers = {
    'step_50_01': step_scheduler_50_01,
    'step_50_05': step_scheduler_50_05,
    'step_20_05': step_scheduler_20_05,
    'step_100_07': step_scheduler_100_07,
    'cosine': cosine_annealing,
}
