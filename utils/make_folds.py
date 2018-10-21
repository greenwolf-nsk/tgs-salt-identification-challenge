import argparse

import numpy as np
import pandas as pd

if __name__ == '__main__':
    np.random.seed(4545)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='../data/train.csv')
    parser.add_argument('-n', '--n_splits', default=10)
    args = parser.parse_args()
    path = args.path
    n_splits = int(args.n_splits)
    train = pd.read_csv(args.path)
    splits = np.repeat(np.arange(n_splits), len(train) // n_splits)
    train['fold'] = np.random.choice(splits, len(splits), replace=False)
    train.to_csv(path.replace('train', 'train_folds'))
