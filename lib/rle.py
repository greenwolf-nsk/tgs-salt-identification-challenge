import numpy as np


def get_mask_rle(mask):
    points = np.where(mask.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for point in points:
        if point > prev + 1:
            run_lengths.extend((point + 1, 0))
        run_lengths[-1] += 1
        prev = point
    return run_lengths
