from utils import SCHEDULER

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import warnings
from copy import deepcopy

@SCHEDULER.register()
class LinearCosineAnnealingLR(_LRScheduler):
    def  __init__(self,
                  optimizer,
                  milestones:list,
                  gamma:float,
                  amplitude:float,
                  cycle:int,
                  cycle_milestones:list,
                  cycle_gamma:float,
                  max_epoch:int,
                  init_lr:float,
                  min_lr = 1e-5,
                  warmup=None,
                  resume=None,
                  ):
        self.optimizer = optimizer
        self.base_lr = init_lr
        if isinstance(self.base_lr, str):
            self.base_lr = eval(self.base_lr)
        self._lr = deepcopy(self.base_lr)
        self.lr_gamma = 1
        self.milestones = milestones
        self.gamma = gamma
        self.amplitude = amplitude
        self.cycle = cycle
        self.cycle_milestones = cycle_milestones
        self.cycle_gamma = cycle_gamma
        if warmup is not None:
            self.eps = warmup-1
        else:
            self.eps = 0
        self.count_down = cycle
        self.increment = self.get_linear_increment(cycle)
        self.max_epoch = max_epoch
        self.min_lr = min_lr
        if isinstance(self.min_lr, str):
            self.min_lr = eval(self.min_lr)
        self.step()

    def get_linear_increment(self, cycle):
        return self.base_lr * self.lr_gamma * self.amplitude / (cycle - 1)

    def get_cosine_increment(self):
        return math.cos((self.eps/self.max_epoch) * math.pi/2) * self._lr + self.min_lr

    def get_lr(self):
        self.eps+=1
        self.count_down -= 1
        self.base_lr = self.get_cosine_increment()
        if self.eps in self.milestones:
            if isinstance(self.gamma, list):
                index = self.milestones.index(self.eps)
                self.lr_gamma = self.lr_gamma * self.gamma[index]
            else:
                self.lr_gamma = self.lr_gamma * self.gamma
        if self.eps in self.cycle_milestones:
            if isinstance(self.cycle_gamma, list):
                index = self.cycle_milestones.index(self.eps)
                self.cycle = self.cycle * self.cycle_gamma[index]
            else:
                self.cycle = self.cycle * self.cycle_gamma
            self.count_down = self.cycle
            self.increment = self.get_linear_increment(self.cycle)
            return self.base_lr * self.lr_gamma
        if self.count_down == 0:
            self.increment = self.get_linear_increment(self.cycle)
            self.count_down = self.cycle
            return self.base_lr * self.lr_gamma
        return self.base_lr * self.lr_gamma + self.increment * self.count_down
    
    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def resume_fucntion(self, resume:int=None):
        for i in range(resume):
            self.get_lr()