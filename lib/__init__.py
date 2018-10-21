from .augmentations import (
    train_augmentations,
    train_augmentations_v2,
    train_augmentations_v3,
    train_augmentations_256,
    val_augmentations_128,
    val_augmentations_resize_128,
    val_augmentations_256,
    crop_resize_from_256,
    crop_from_128,
    resize_from_any
)

augs = {
    'train_128_v1': train_augmentations,
    'train_128_v2': train_augmentations_v2,
    'train_128_v3': train_augmentations_v3,
    'train_256_v1': train_augmentations_256,
    'val_128': val_augmentations_128,
    'val_128_resize': val_augmentations_resize_128,
    'val_256': val_augmentations_256
}

transforms = {
    'crop_resize_from_256': crop_resize_from_256,
    'crop_from_128': crop_from_128,
    'resize_from_any': resize_from_any,
}