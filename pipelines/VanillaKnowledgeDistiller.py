from utils import DISTILLER

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

TS={"Epoch":200, "ALPHA":0.5, "BETA":0.5}

@DISTILLER.register()
class VanillaKnowledgeDistiller():
    def __init__(self, 
                 teacher, 
                 device,
                 kd_loss_fucntion, 
                 ALPHA=0.5, BETA=0.5, 
                 temperature=4,
                 NameAbbr=False):
        self.teacher = teacher.to(device)
        self.teacher.eval()
        self.kd_loss_fucntion = kd_loss_fucntion
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.temperature = temperature
        if NameAbbr:
            self.loss_name = "VLoss"
        else:
            self.loss_name = "VanillaKDLoss"

    def distill(self, data, labels, student):
        s_pred = student(data)
        t_pred = self.teacher(data)
        s_pred = s_pred[-1]/self.temperature
        #s_pred = self.SOFTMAX(s_pred[-1])
        t_pred = self.SOFTMAX(t_pred[-1])
        kd_loss = self.kd_loss_fucntion(s_pred, t_pred) * self.BETA
        return self.logger.log(kd_loss, self.loss_name)

    def SOFTMAX(self, x):
        return nn.functional.softmax(x/self.temperature, dim=1)

    def init_logger(self, logger):
        self.logger = logger