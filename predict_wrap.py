import os
import logging
from typing import List

import click
import numpy as np
import pandas as pd
from albumentations.augmentations.functional import center_crop, hflip, vflip

from lib.logger import configure_logger
from lib.image import get_mask_rle
from common import DATA_DIR, SUBMISSION_DIR, get_current_datetime


def wrap_border(mask_src: np.ndarray) -> np.ndarray:
    mask = mask_src.copy()
    mask[14: 27, :] = (mask[14: 27, :] + vflip(mask[:13, :])) / 2
    mask[:, 14: 27] = (mask[:, 14: 27] + hflip(mask[:, :13])) / 2
    mask[99: 113, :] = (mask[99: 113, :] + vflip(mask[114:, :])) / 2
    mask[:, 99: 113] = (mask[:, 99: 113] + hflip(mask[:, 114:])) / 2
    return center_crop(mask, 101, 101)


@click.command()
@click.option('--mask_fn')
@click.option('--fn', default='submission')
@click.option('--th', default=0.5)
def wrap_masks(mask_fn: str, fn: str, th: float):
    logger = configure_logger('prediction', logging.INFO, './logs')
    test = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    masks = np.load(mask_fn)

    np.save(os.path.join(SUBMISSION_DIR, 'avg_masks'), masks)
    masks = [(wrap_border(mask) > th).astype(int) for mask in masks]
    rle_masks = [get_mask_rle(mask) for mask in masks]
    test['rle_mask'] = rle_masks
    submission_fn = os.path.join(SUBMISSION_DIR, f'{fn}_{get_current_datetime()}.csv')
    test.to_csv(submission_fn, index=None)
    logger.info(f'Saved submission to {submission_fn}')


if __name__ == '__main__':
    wrap_masks()