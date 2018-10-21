import os
import logging

import tqdm
import torch
import click
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from albumentations.augmentations.functional import center_crop, hflip

from net import architectures
from lib import NET_INPUT_SIZE, SRC_SIZE
from lib.dataloader import SaltTestDataset, val_augmentations, flip_pad
from lib.logger import configure_logger
from lib.image import get_mask_rle
from common import DATA_DIR, SNAPSHOT_DIR, SUBMISSION_DIR, get_current_datetime


@click.command()
@click.option('--snapshot_dir', default='snapshots')
@click.option('--arch', default='unet_resnet34_v1')
@click.option('--device', default='cuda:0')
@click.option('--fn', default='submission')
@click.option('--th', default=0.5)
@click.option('--masks_fn', default=None)
def predict_with_snapshot(snapshot_dir: str, arch: str, device: str, fn: str, th: float, masks_fn: str):
    logger = configure_logger('prediction', logging.INFO, './logs')
    test = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    if masks_fn is None:
        test_images = np.load(os.path.join(DATA_DIR, 'test_images.npy'))
        test_ids = test.index.tolist()
        model = architectures[arch]().to(device)
        test = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
        test_images = np.load(os.path.join(DATA_DIR, 'test_images.npy'))
        test_ids = test.index.tolist()
        tta_augs = [val_augmentations, flip_pad]
        tta_predictions = []
        for cycle_dir in os.listdir(snapshot_dir):
            snapshots = os.listdir(os.path.join(snapshot_dir, cycle_dir))
            best_snapshot = sorted(snapshots, key=lambda x: int(x.split('.')[-1]), reverse=True)[0]

            state_dict = torch.load(os.path.join(snapshot_dir, cycle_dir, best_snapshot))

            model.load_state_dict(state_dict)
            model.eval()
            logger.info(f'Loaded model from {best_snapshot}')

            for i, aug in enumerate(tta_augs):
                test_dataset = SaltTestDataset(test_ids, test_images, aug)
                test_loader = DataLoader(test_dataset, 30, shuffle=False)

                # actual prediction is made here
                masks = []
                with torch.no_grad():
                    for batch in tqdm.tqdm(test_loader):
                        image = batch['image'].to(device)
                        y_pred = torch.sigmoid(model(image)).cpu().numpy()
                        masks.append(y_pred)

                # postprocess masks (crop, threshold, rle)
                masks = np.concatenate(masks).reshape((len(test), NET_INPUT_SIZE, NET_INPUT_SIZE))
                # TODO: replace that with smth that makes more sens
                if i == 1:
                    masks = [hflip(mask) for mask in masks]
                tta_predictions.append(masks)

        masks = np.mean(tta_predictions, axis=0)
        np.save(os.path.join(SUBMISSION_DIR, 'raw_masks_fold1_scnd.npy'), masks)
    else:
        masks = np.load(os.path.join(SUBMISSION_DIR, masks_fn))

    masks = [(center_crop(mask, SRC_SIZE, SRC_SIZE) > th).astype(int) for mask in masks]
    rle_masks = [get_mask_rle(mask) for mask in masks]
    test['rle_mask'] = rle_masks
    # TODO: get some stats on empty masks etc. there and log it too
    submission_fn = os.path.join(SUBMISSION_DIR, f'{fn}_{get_current_datetime()}.csv')
    test.to_csv(submission_fn, index=None)
    logger.info(f'Saved submission to {submission_fn}')


if __name__ == '__main__':
    predict_with_snapshot()