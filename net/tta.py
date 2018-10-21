from typing import Callable

import numpy as np
from albumentations.augmentations.functional import hflip


class TTA:

    def __init__(self, transform_fn: Callable, inverse_transform_fn: Callable):
        self.transform_fn = transform_fn
        self.inverse_transform_fn = inverse_transform_fn

    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Apply transform to images
        :param images: n_images x height x width array
        :return: tta_images: n_images x height x width array
        """
        return np.array([self.transform_fn(image) for image in images])

    def inverse_transform(self, masks: np.ndarray) -> np.ndarray:
        """
        Apply inverse_transform to masks
        :param masks: n_images x height x width array
        :return: tta_masks: n_images x height x width array
        """
        return np.array([self.inverse_transform_fn(mask) for mask in masks])


null_tta = TTA(lambda x: x, lambda x: x)
hflip_tta = TTA(hflip, hflip)

tta_dict = {
    'null_hflip': [null_tta, hflip_tta],
}
