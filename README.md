# TGS Salt Identification Challenge solution
This repository contatins solution code for https://www.kaggle.com/c/tgs-salt-identification-challengewww.kaggle.com/c/tgs-salt-identification-challenge

This is my first time writing pipeline for deep learning competition, so there might be some non-optimal coding decisions.

### Data preparation
Before training the model you will need to use two scripts from utils directory.

First one stores all train/test images and train masks in single numpy array, 
it makes further work with data much easier, though it won't work for datasets larger than your RAM.
```bash
python3 stack_and_save.py --data_dir path/to/unzipped/data
```
Another script creates train_with_folds.csv, which adds 'fold' column to train.csv
Folds are split randomly (non-stratified)
```bash
python3 make_folds.py --p path/to/train.csv --n 10
```


### Model training
Model could be trained using [train.py](train.py), which supports different training parameters,
including model architecture, augmentations and different optimizers / schedulers

```bash
python3 train.py --arch wide_resnet_v1 --batch_size=32 --loss BCE --optimizer adam_1e-4 --experiment_dir experiments/wide_fold0_128 --n_epochs 200 --train_augs train_128_v1 --val_augs val_128_resize --val_transforms resize_from_any  --fold 0
```

By default, snapshots and predicted validations masks are saved if any of used metrics improved from previous snapshot.

### Making prediction
After training the model, you could take one of the snapshots to make a prediction.
Before that you may want to find best treshold as it is not always 0.5
```bash
python3 optimize_threshold.py --masks_fn experiments/wide_fold0/snapshots/val_masks_wide_fold0_resize_ft/snapshots/epoch_194.loss_1.1063.iout_0.855.npy
```

```bash
python3 predict.py --snapshot experiments/wide_fold0/snapshots/epoch_194.loss_1.1063.iout_0.855 --th 0.51 --arch wide_resnet_v1
```

Raw (not cropped / thresholded) masks are also in predict.py

### Other options
There are some other scripts apart from mentioned above.

[predict_tta.py](predict_tta.py) - make prediction with TTA (test time augmentation), TTA options are defined [there](./net/tta.py)

[predict_se.py](predict_se.py) - make prediction from best submissions from snapshot ensembling (need to train with --cycles > 1)

[predict_fold_avg.py](predict_fold_avg.py) - simply average all masks from files passed in arguments

[train_border.py](train_border.py) - script to train net with mask and border outputs. 
It was last day experiment, so I didn't have time to include this logic into main pipeline.
