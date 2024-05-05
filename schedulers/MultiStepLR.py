from utils import SCHEDULER

from torch.optim import lr_scheduler

@SCHEDULER.register()
def MultiStepLR(optimizer, **kwargs):
    return lr_scheduler.MultiStepLR(optimizer=optimizer, **kwargs)