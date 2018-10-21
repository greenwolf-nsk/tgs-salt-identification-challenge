import numpy as np

from albumentations import *
from albumentations.augmentations.functional import resize, center_crop

train_augmentations = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        ShiftScaleRotate(rotate_limit=15),
        ElasticTransform(),
        GridDistortion(),
    ], p=0.5),
    OneOf([
        RandomBrightness(limit=0.1),
        RandomGamma(),
        RandomContrast(),
    ], p=0.5),
    PadIfNeeded(min_height=128, min_width=128),
    Normalize()
])


train_augmentations_v2 = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        ShiftScaleRotate(rotate_limit=10, scale_limit=0, shift_limit=0.1),
    ], p=0.5),
    OneOf([
        RandomBrightness(limit=0.1),
    ], p=0.5),
    PadIfNeeded(min_height=128, min_width=128),
    Normalize()
])

train_augmentations_256 = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        ShiftScaleRotate(rotate_limit=10, scale_limit=0.3, shift_limit=0.1),
    ], p=0.5),
    OneOf([
        RandomBrightness(limit=0.1),
        RandomGamma(),
        RandomContrast(),
    ], p=0.5),
    Resize(202, 202),
    PadIfNeeded(min_height=256, min_width=256),
    Normalize()
])

train_augmentations_v3 = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        ShiftScaleRotate(rotate_limit=10, scale_limit=0.3, shift_limit=0.1),
        RandomSizedCrop((32, 96), 101, 101),
    ], p=0.5),
    OneOf([
        RandomBrightness(limit=0.1),
        RandomGamma(),
        RandomContrast(),
    ], p=0.5),
    OneOf([
        Blur(),
        MedianBlur(),
        GaussNoise(),
    ], p=0.5),
    Resize(128, 128),
])

val_augmentations_128 = Compose([
    PadIfNeeded(min_height=128, min_width=128),
    Normalize()
])

val_augmentations_resize_128 = Compose([
    Resize(128, 128),
    Normalize()
])

val_augmentations_256 = Compose([
    Resize(202, 202),
    PadIfNeeded(min_height=256, min_width=256),
    Normalize()
])


def crop_resize_from_256(image: np.ndarray) -> np.ndarray:
    cropped = center_crop(image, 202, 202)
    return resize(cropped, 101, 101)


def crop_from_128(image: np.ndarray) -> np.ndarray:
    return center_crop(image, 101, 101)


def resize_from_any(image: np.ndarray) -> np.ndarray:
    return resize(image, 101, 101)



















