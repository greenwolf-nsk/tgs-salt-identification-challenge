from functools import partial

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

step_scheduler_50_01 = partial(StepLR, step_size=50, gamma=0.1)
step_scheduler_50_05 = partial(StepLR, step_size=50, gamma=0.5)
step_scheduler_20_05 = partial(StepLR, step_size=20, gamma=0.5)
step_scheduler_100_07 = partial(StepLR, step_size=100, gamma=0.7)

cosine_annealing = partial(CosineAnnealingLR, T_max=50, eta_min=0.001)
