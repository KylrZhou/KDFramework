from utils import OPTIMIZER

import torch.optim as optim

@OPTIMIZER.register()
def SGD(model_params, **kwargs):
    return optim.SGD(model_params, **kwargs)