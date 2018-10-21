import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SaltTrainDataset(Dataset):

    def __init__(self, image_ids: np.ndarray, images: np.ndarray, masks: np.ndarray = None, augs=None):
        self.image_ids = image_ids
        self.images = images[image_ids]
        self.masks = masks[image_ids]
        self.augs = augs

    def __getitem__(self, idx: int):
        image, mask = self.images[idx], self.masks[idx]
        augs = self.augs(image=image, mask=mask)
        image, mask = augs['image'], augs['mask'] / 255

        return {
            'image': torch.Tensor(image.transpose(2, 0, 1)),
            'mask': torch.Tensor(mask).unsqueeze(0)
        }

    def __len__(self):
        return len(self.images)


class SaltTestDataset(Dataset):

    def __init__(self, image_ids: np.ndarray, images: np.ndarray, augs=None):
        self.image_ids = image_ids
        self.images = images[image_ids]
        self.augs = augs

    def __getitem__(self, idx: int):
        image = self.images[idx]
        augs = self.augs(image=image)
        image = augs['image']

        return {
            'image': torch.Tensor(image.transpose(2, 0, 1)),
        }

    def __len__(self):
        return len(self.images)


def find_contours(mask: np.array):
    full_image = np.ones((101, 101), dtype=np.uint8)
    _, full_contours, _ = cv2.findContours(full_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_contours = cv2.drawContours(np.zeros((101, 101), dtype=np.uint8), contours, -1, (255, 255, 255), 1)
    full_contours = cv2.drawContours(np.zeros((101, 101), dtype=np.uint8), full_contours, -1, (255, 255, 255), 1)
    return np.logical_and(mask_contours, np.logical_not(full_contours)) * 255


class SaltTrainDatasetBoundary(Dataset):

    def __init__(self, image_ids: np.ndarray, images: np.ndarray, masks: np.ndarray = None, augs=None):
        self.image_ids = image_ids
        self.images = images[image_ids]
        self.masks = masks[image_ids]
        self.augs = augs

    def __getitem__(self, idx: int):
        image, mask = self.images[idx], self.masks[idx]
        border_mask = find_contours(mask)
        # in order to make this work you'll need to fix one line in albumentations library
        augs = self.augs(image=image, mask=mask, border_mask=border_mask)
        image, mask = augs['image'], np.array([augs['mask'], augs['border_mask']]) / 255

        return {
            'image': torch.Tensor(image.transpose(2, 0, 1)),
            'mask': torch.Tensor(mask)
        }

    def __len__(self):
        return len(self.images)
