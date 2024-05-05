import os
import sys
import re
import datetime

import yaml
import numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms

from augmentations import *
from datasets import *
from models import *
from optimizers import *
from pipelines import *
from schedulers import *
from utils import *
from functions import *

#My scheduler
#My augmentation
#resume training
#my method adaptation
if __name__ == '__main__':
    config = yaml.load(open('configs/base_training.yaml', 'r'),  Loader=yaml.FullLoader)
    
    train_dataset = build_dataset_train(config['train_dataset'])
    test_dataset = build_dataset_val(config['test_dataset'])
    
    model = build_model(config['model'])
    model = model.to(config['settings']['device'])
    
    loss_func = build_fucntion(config['settings']['loss_function'])

    optimizer = build_optimizer(model.parameters(), config['settings']['optimizer'])
    scheduler = build_scheduler(optimizer, config['settings']['scheduler'])
    
    logger = build_logger(config['logger'])
    logger.init_dataset(train_dataset)
    logger.init_model(model)
    
    try:
        distiller = build_distiller(config['distiller'])
        distiller.init_logger(logger)
        KDBaseTrain(train_dataset=train_dataset, 
                    test_dataset=test_dataset, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    model=model, 
                    loss_function=loss_func, 
                    logger=logger, 
                    config=config,
                    distiller=distiller)
    except KeyError:
        distiller = None
        BaseTrain(train_dataset=train_dataset, 
                  test_dataset=test_dataset, 
                  optimizer=optimizer, 
                  scheduler=scheduler, 
                  model=model, 
                  loss_function=loss_func, 
                  logger=logger, 
                  config=config)