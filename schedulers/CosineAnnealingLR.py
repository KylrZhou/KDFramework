from utils import SCHEDULER

from torch.optim import lr_scheduler

@SCHEDULER.register()
def CosineAnnealingLR(optimizer, **kwargs):
    return lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **kwargs)