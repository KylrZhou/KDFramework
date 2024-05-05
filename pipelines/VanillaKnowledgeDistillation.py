import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from utils import PIPELINE

TS={"Epoch":200, "ALPHA":0.5, "BETA":0.5}

@PIPELINE.register()
def VanillaKnowledgeDistillation(
    TrainSettings:dict,
    TrainLoader:DataLoader,
    ValLoader:DataLoader,
    StudentNetwork:nn.Module,
    TeacherNetwork:nn.Module,
    LabelLossFunction,
    KDLossFunction,
    AccFunction,
    Optimizer:Optimizer,
    Scheduler:_LRScheduler,
    Logger=None):
    ALPHA = TrainSettings['ALPHA']
    BETA = TrainSettings['BETA']
    if KDLossFunction is None:
        KDLossFunction = LabelLossFunction
    for epoch in range(1, TrainSettings['Epoch']+1):
        for batch_idx, (data, label) in enumerate(TrainLoader):
            s_pred = StudentNetwork(data)
            t_pred = TeacherNetwork(data)
            if isinstance(s_pred, list):
                s_pred = s_pred[-1]
            if isinstance(t_pred, list):
                t_pred = t_pred[-1]
            lable_loss = LabelLossFunction(s_pred, label)
            kd_loss = KDLossFunction(s_pred, t_pred)
            Optimizer.zero_grad()
            loss = label_loss *ALPHA + kd_loss *BETA
            loss.backward()
            logger.log()
        Scheduler.step()
        for batch_idx, (data, label) in enumerate(ValLoader):
            s_pred = StudentNetwork(data)
            t_pred = TeacherNetwork(data)
            if isinstance(s_pred, list):
                s_pred = s_pred[-1]
            if isinstance(t_pred, list):
                t_pred = t_pred[-1]
            s_acc = AccFunction(s_pred, label)
            t_acc = AccFunction(t_pred, label)
            logger.log()