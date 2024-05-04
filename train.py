import os
import sys
import re
import datetime

import numpy


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import torchvision
import torchvision.transforms as transforms

from models import MobileNetV2
from utlis import Logger

def get_training_dataloader(mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404), batch_size=384, num_workers=12, shuffle=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404), batch_size=384, num_workers=12, shuffle=False):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

if __name__ == '__main__':
    train_loader = get_training_dataloader()
    test_loader = get_test_dataloader()
    logger = Logger(log_interval=50, MAX_ITER=len(train_loader), checkpoint_interval=2)
    model = MobileNetV2()
    model = model.to('cuda')
    logger.init_model(model)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.1)
    
    logger.data_time_start()
    for epoch in range(1,201):
        for batch_idx, (image, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            image = image.to('cuda')
            labels = labels.to('cuda')
            logger.data_time_end()
            logger.calc_time_start()
            outputs = model(image)
            outputs = outputs[-1]
            loss = logger.log(loss_func(outputs, labels), "Loss")
            loss.backward()
            logger.calc_time_end()
            logger.data_time_start()
            n_iter = batch_idx + 1
            lr = logger.log(optimizer.param_groups[0]['lr'], "lr")
            log_dict = {"Loss":loss,"lr":lr}
            logger.update()
        correct = 0
        logger.calc_time_start()
        for batch_idx, (image, labels) in enumerate(test_loader):
            image = image.to('cuda')
            labels = labels.to('cuda')
            
            outputs = model(image)
            outputs = outputs[-1]
            
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
        logger.calc_time_end()
        logger.log_val(correct.float()/len(test_loader.dataset), "Acc")
        logger.update_val()
        logger.data_time_start()
        #print(f"Epoch: {epoch}\t Acc: {correct.float()/len(test_loader.dataset)}")