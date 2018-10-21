import os
import json
import warnings
from typing import Callable, Dict
from collections import defaultdict

import torch
import click
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

from net import architectures, losses, optimizers, schedulers
from lib.dataloader import SaltTrainDataset
from lib import augs, transforms
from lib.metrics import batch_iou, batch_iout
from lib.logger import ExperimentLogger
from common import DATA_DIR, SNAPSHOT_DIR, LOG_DIR
warnings.filterwarnings('ignore')


@click.command()
@click.option('--experiment_dir', default='experiments')
@click.option('--arch', default='unet_resnet34_v1', type=click.Choice(architectures))
@click.option('--snapshot', default=None)
@click.option('--train_augs', default='train_128_v1')
@click.option('--val_augs', default='val_128')
@click.option('--val_transforms', default='crop_from_128')
@click.option('--fold', default=0)
@click.option('--loader_workers', default=4)
@click.option('--optimizer', default='adam_1e-4', type=click.Choice(optimizers))
@click.option('--loss', default='BCE', type=click.Choice(losses))
@click.option('--scheduler', default=None, type=click.Choice(schedulers))
@click.option('--cycles', default=1)
@click.option('--batch_size', default=32)
@click.option('--n_epochs', default=100)
@click.option('--device', default='cuda:0')
@click.option('--th', default=0.0)
def train_cli(
        experiment_dir: str,
        arch: str,
        train_augs: str,
        val_augs: str,
        val_transforms: str,
        snapshot: str,
        fold: int,
        loader_workers: int,
        optimizer: str,
        loss: str,
        scheduler: str,
        batch_size: int,
        n_epochs: int,
        device: str,
        cycles: int,
        th: float,
):
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        os.makedirs(os.path.join(experiment_dir, LOG_DIR))
        os.makedirs(os.path.join(experiment_dir, SNAPSHOT_DIR))

    params = locals()
    with open(os.path.join(experiment_dir, 'train_config.json'), 'w') as f:
        json.dump(params, f, indent=4)

    log_field_names = ['epoch', 'train_loss', 'train_iou', 'train_iout', 'val_loss', 'val_iou', 'val_iout']
    logger = ExperimentLogger('train', os.path.join(experiment_dir, LOG_DIR), log_field_names)

    snapshot_exists = snapshot is not None
    model = architectures[arch](pretrained=not snapshot_exists).to(device)
    if snapshot_exists:
        state_dict = torch.load(snapshot)
        model.load_state_dict(state_dict, strict=False)

    loss_fn = losses[loss]

    # prepare train and val datasets
    train_with_folds = pd.read_csv('../data/train_folds.csv')
    train_images = np.load(os.path.join(DATA_DIR, 'train_images.npy'))
    train_masks = np.load(os.path.join(DATA_DIR, 'train_masks.npy'))
    train_ids = train_with_folds.query(f'fold != {fold}').index.tolist()
    val_ids = train_with_folds.query(f'fold == {fold}').index.tolist()
    train_dataset = SaltTrainDataset(train_ids, train_images, train_masks, augs[train_augs])
    val_dataset = SaltTrainDataset(val_ids, train_images, train_masks, augs[val_augs])

    if cycles == 1:
        optimizer_ = optimizers[optimizer](model.parameters())
        scheduler_ = schedulers[scheduler](optimizer_) if scheduler else None

        train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            val_transforms=transforms[val_transforms],
            loss_fn=loss_fn,
            metrics={'iou': batch_iou, 'iout': batch_iout},
            batch_size=batch_size,
            optimizer=optimizer_,
            scheduler=scheduler_,
            logger=logger,
            n_epochs=n_epochs,
            loader_workers=loader_workers,
            snapshot_dir=os.path.join(experiment_dir, SNAPSHOT_DIR)
        )
    else:
        for cycle in range(cycles):
            optimizer_ = optimizers[optimizer](model.parameters())
            scheduler_ = schedulers[scheduler](optimizer_) if scheduler else None

            train_model(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                val_transforms=transforms[val_transforms],
                loss_fn=loss_fn,
                metrics={'iou': batch_iou, 'iout': batch_iout},
                batch_size=batch_size,
                optimizer=optimizer_,
                scheduler=scheduler_,
                logger=logger,
                n_epochs=n_epochs,
                loader_workers=loader_workers,
                snapshot_dir=os.path.join(experiment_dir, SNAPSHOT_DIR),
                cycle=cycle
            )


def train_model(
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        val_transforms: Callable,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        loader_workers: int = 4,
        batch_size: int = 32,
        n_epochs: int = 20,
        device: str = 'cuda:0',
        logger: ExperimentLogger = None,
        snapshot_dir: str = './snapshots',
        cycle: int = None,
):
    # TODO: adaptive learning rate, early stopping conditions
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=loader_workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=loader_workers)
    if cycle is not None:
        snapshot_dir = os.path.join(snapshot_dir, f'cycle_{cycle}')
        os.makedirs(snapshot_dir)
    best_mertics_dict = {metric_name: 0 for metric_name in metrics}
    best_val_loss = 1e10
    for epoch in range(n_epochs):
        metrics_dict = defaultdict(list)

        train_loss = []
        val_loss = []
        model.train(True)
        for batch in train_loader:
            image, mask = batch['image'], batch['mask']
            image = image.to(device)
            y_pred = model(image)
            loss = loss_fn(y_pred, mask.to(device))
            for metric_name, metric in metrics.items():
                metrics_dict[f'train_{metric_name}'] += metric(y_pred, mask.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())

        model.eval()
        orig_masks = []
        pred_masks = []
        e_loss = []
        for batch in val_loader:
            image, mask_raw = batch['image'], batch['mask']
            image = image.to(device)
            with torch.no_grad():
                y_pred_raw = model(image)
                y_pred = torch.sigmoid(y_pred_raw).cpu().squeeze().numpy()
                mask = mask_raw.squeeze().numpy()
                e_loss.append(loss_fn(y_pred_raw, mask_raw.to(device)))
                for i in range(mask.shape[0]):
                    predicted_mask = y_pred
                    original_mask = mask
                    pred_masks.append(val_transforms(predicted_mask.squeeze()))
                    orig_masks.append(val_transforms(original_mask.squeeze()))

        pred_masks_numpy = np.array(pred_masks)
        orig_masks = torch.Tensor(orig_masks).unsqueeze(1)
        pred_masks = torch.Tensor(pred_masks_numpy).unsqueeze(1)
        val_loss.append(np.mean(e_loss))

        for metric_name, metric in metrics.items():
            metrics_dict[f'val_{metric_name}'] += metric(pred_masks, orig_masks, th=0.5)

        if scheduler:
            scheduler.step(epoch=epoch)

        logger.log(
            epoch=epoch,
            train_loss=np.mean(train_loss),
            train_iou=np.mean(metrics_dict['train_iou']),
            train_iout=np.mean(metrics_dict['train_iout']),
            val_loss=np.mean(val_loss),
            val_iou=np.mean(metrics_dict['val_iou']),
            val_iout=np.mean(metrics_dict['val_iout']),
            lr=optimizer.param_groups[0]['lr'],
        )
        logger.log_val_metrics(','.join([f'{x:.4f}' for x in metrics_dict['val_iou']]))

        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            snapshot_name = f'epoch_{epoch}.loss_{np.mean(val_loss):.4f}.iout_{np.mean(metrics_dict["val_iout"]):.3f}'
            np.save(arr=pred_masks_numpy, file=os.path.join(snapshot_dir, 'val_masks_' + snapshot_name))
            torch.save(model.state_dict(), os.path.join(snapshot_dir, snapshot_name))

        for metric_name in metrics:
            if best_mertics_dict[metric_name] < np.mean(metrics_dict[f'val_{metric_name}']):
                best_mertics_dict[metric_name] = np.mean(metrics_dict[f'val_{metric_name}'])
                snapshot_name = f'epoch_{epoch}.loss_{np.mean(val_loss):.4f}.iout_{np.mean(metrics_dict["val_iout"]):.3f}'
                np.save(arr=pred_masks_numpy, file=os.path.join(snapshot_dir, 'val_masks_' + snapshot_name))
                torch.save(model.state_dict(), os.path.join(snapshot_dir, snapshot_name))


if __name__ == '__main__':
    train_cli()
