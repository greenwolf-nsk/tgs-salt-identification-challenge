import os
import logging

import tqdm
import torch
import click
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from net import architectures
from net.tta import tta_dict
from lib import augs, transforms
from lib.dataloader import SaltTestDataset
from lib.logger import configure_logger
from lib.image import get_mask_rle
from common import DATA_DIR, SNAPSHOT_DIR, SUBMISSION_DIR, get_current_datetime


@click.command()
@click.option('--snapshot', default='best_model.pt')
@click.option('--arch', default='unet_resnet34_v1')
@click.option('--val_augs', default='val_128')
@click.option('--transform', default='crop_from_128')
@click.option('--tta', default='null_hflip')
@click.option('--device', default='cuda:0')
@click.option('--masks_fn', default='masks.npy')
@click.option('--fn', default='submission')
@click.option('--th', default=0.5)
def predict_with_snapshot(
        snapshot: str,
        arch: str,
        val_augs: str,
        transform: str,
        tta: str,
        device: str,
        fn: str,
        masks_fn: str,
        th: float
):
    logger = configure_logger('prediction', logging.INFO, './logs')

    # load model from snapshot
    state_dict = torch.load(snapshot)
    model = architectures[arch]().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f'Loaded model from {snapshot}')

    # load test data
    test = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    test_images = np.load(os.path.join(DATA_DIR, 'test_images.npy'))
    test_ids = test.index.tolist()

    # actual prediction is made here
    tta_masks = []
    for tta_transformer in tta_dict[tta]:
        tta_images = tta_transformer.transform(test_images)
        test_dataset = SaltTestDataset(test_ids, tta_images, augs[val_augs])
        test_loader = DataLoader(test_dataset, 30, shuffle=False)
        masks = []
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                image = batch['image'].to(device)
                y_pred = torch.sigmoid(model(image)).cpu().numpy()
                masks.append(y_pred)
            # postprocess masks (crop, threshold, rle)
            height, width = masks[0][0][0].shape
        masks = np.concatenate(masks).reshape((len(test), height, width))
        masks = tta_transformer.inverse_transform(masks)
        tta_masks.append(masks)

    masks = np.mean(tta_masks, axis=0)
    np.save(os.path.join(SUBMISSION_DIR, masks_fn), masks)

    masks = [(transforms[transform](mask) > th).astype(int) for mask in masks]
    rle_masks = [get_mask_rle(mask) for mask in masks]
    test['rle_mask'] = rle_masks
    # TODO: get some stats on empty masks etc. there and log it too
    submission_fn = os.path.join(SUBMISSION_DIR, f'{fn}_{get_current_datetime()}.csv')
    test.to_csv(submission_fn, index=None)
    logger.info(f'Saved submission to {submission_fn}')


if __name__ == '__main__':
    predict_with_snapshot()
