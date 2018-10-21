from functools import partial

from torch.optim import Adam, SGD


adam_1e3 = partial(Adam, lr=1e-3)
adam_1e4 = partial(Adam, lr=1e-4)
adam_1e5 = partial(Adam, lr=1e-5)

sgd_1e2 = partial(SGD, lr=1e-2, momentum=0.01, weight_decay=0.001)
sgd_1e3 = partial(SGD, lr=1e-3, momentum=0.01, weight_decay=0.001)
sgd_1e4 = partial(SGD, lr=1e-4, momentum=0.01, weight_decay=0.001)
