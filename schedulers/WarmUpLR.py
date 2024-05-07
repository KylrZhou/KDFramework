from utils import SCHEDULER

from torch.optim.lr_scheduler import _LRScheduler

@SCHEDULER.register()
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, iter_per_epoch, warmup_epochs, last_epoch=-1):

        self.warmup_epochs = warmup_epochs
        self.total_iters = iter_per_epoch * warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
