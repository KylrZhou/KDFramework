from utils import FUNCTION

import torch.nn as nn

@FUNCTION.register()
def CrossEntropyLoss(**kwargs):
    return nn.CrossEntropyLoss(**kwargs)

@FUNCTION.register()
def KLDivLoss(**kwargs):
    return nn.KLDivLoss(**kwargs)