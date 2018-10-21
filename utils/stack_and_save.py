import os
import click

import cv2
import pandas as pd
import numpy as np


def load_and_stack(image_ids: list, data_dir: str, suffix: str = 'images') -> np.ndarray:
    images = []
    for image_id in image_ids:
        img_path = os.path.join(data_dir, f'{suffix}/{image_id}.png')
        img = cv2.imread(img_path)
        if suffix == 'images':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            images.append(img[:, :, 0])

    return np.array(images)


@click.command()
@click.option('--data_dir', default='../data')
def stack_and_save(data_dir: str):
    train_ids = pd.read_csv(os.path.join(data_dir, 'train.csv'))['id'].values
    test_ids = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))['id'].values
    train_images = load_and_stack(train_ids, data_dir, 'images')
    train_masks = load_and_stack(train_ids, data_dir, 'masks')
    test_images = load_and_stack(test_ids, data_dir, 'images')
    np.save(os.path.join(data_dir, f'train_images.npy'), train_images)
    np.save(os.path.join(data_dir, f'train_masks.npy'), train_masks)
    np.save(os.path.join(data_dir, f'test_images.npy'), test_images)
    print(f'saved train & test!')
        

if __name__ == '__main__':
    stack_and_save()
