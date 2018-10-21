import os
import logging

import torch
import click
import pandas as pd
import numpy as np

from lib.metrics import batch_iout
from lib.logger import configure_logger
from common import DATA_DIR


@click.command()
@click.option('--fold', default=0)
@click.option('--masks_fn', default=None)
def optimize_thresholds(fold: int, masks_fn: str):
    logger = configure_logger('prediction', logging.INFO, './logs')

    # prepare train and val datasets
    train_with_folds = pd.read_csv('../data/train_folds.csv')
    train_masks = np.load(os.path.join(DATA_DIR, 'train_masks.npy'))
    val_ids = train_with_folds.query(f'fold == {fold}').index.tolist()

    val_masks = torch.Tensor(train_masks[val_ids]).unsqueeze(1) / 255.
    pred_masks = torch.Tensor(np.load(masks_fn)).unsqueeze(1)
    best_th_so_far = 0
    best_iout_so_far = 0

    for th in np.linspace(0, 1, 101):
        iout = np.mean(batch_iout(pred_masks, val_masks, th))
        if iout > best_iout_so_far:
            best_iout_so_far = iout
            best_th_so_far = th

        logger.info(f'{th:^6.3f}|{np.mean(iout):^8.6f}')

    logger.info(f'Best threshold: {best_th_so_far:^6.3f}; iout: {best_iout_so_far:^8.6f}')


if __name__ == '__main__':
    optimize_thresholds()
