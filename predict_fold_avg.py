import os
import logging
import argparse
from typing import List

import numpy as np
import pandas as pd
from albumentations.augmentations.functional import center_crop, hflip

from lib.logger import configure_logger
from lib.rle import get_mask_rle
from common import DATA_DIR, SUBMISSION_DIR, get_current_datetime


def avg_masks(mask_fns: List[str], fn: str, th: float):
    logger = configure_logger('prediction', logging.INFO, './logs')
    test = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    masks = []
    for mask_fn in mask_fns:
        masks.append(np.load(mask_fn))
    masks = np.mean(masks, axis=0)

    np.save(os.path.join(SUBMISSION_DIR, 'avg_masks_fold_0256'), masks)
    masks = [(center_crop(mask, 101, 101) > th).astype(int) for mask in masks]
    rle_masks = [get_mask_rle(mask) for mask in masks]
    test['rle_mask'] = rle_masks
    submission_fn = os.path.join(SUBMISSION_DIR, f'{fn}_{get_current_datetime()}.csv')
    test.to_csv(submission_fn, index=None)
    logger.info(f'Saved submission to {submission_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mask_files', nargs='+')
    parser.add_argument('--th', default=0.5)
    parser.add_argument('--fn', default='submission')
    args = parser.parse_args()

    avg_masks(args.mask_files, args.fn, args.th)
