import os
import sys
import re
import datetime

import argparse
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

#cuda device problem implant to dataset
#! debuging first
#resume training (MY scheduler problem)
# repair Vanilla pipeline
#! My scheduler justification 1 epoch more problem
#my method adaptation
###constructing into one file
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_training.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    resume = None
    if args.resume is not None:
        resume = torch.load(args.resume)

    config = yaml.load(open(args.config, 'r'),  Loader=yaml.FullLoader)
    """
    config['settings']['EPOCHS'] = 1
    config['logger']['Write2File'] = False
    config['logger']['SaveCheckpoint'] = False
    config['logger']['Upload2Wandb'] = False
    """
    
    config['filename'] = extract_filename(args.config)
     
    train_dataset = build_dataset_train(config['train_dataset'])
    test_dataset = build_dataset_val(config['test_dataset'])
    
    model = build_model(config['model'])
    model = model.to(config['settings']['device'])
    
    loss_func = build_function(config['settings']['loss_function'])

    optimizer = build_optimizer(model.parameters(), config['settings']['optimizer'])
    scheduler = build_scheduler(optimizer, config['settings']['scheduler'])
    try:
        config['settings']['warmup']['iter_per_epoch'] = len(train_dataset)
        warmup = build_scheduler(optimizer, config['settings']['warmup'])
    except:
        warmup = None
    logger = build_logger(config)
    logger.init_dataset(train_dataset)
    logger.init_model_optimizer_scheduler(model, optimizer, scheduler)
    
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
                    warmup=warmup,
                    config=config,
                    distiller=distiller,
                    resume=resume)
    except KeyError:
        distiller = None
        BaseTrain(train_dataset=train_dataset, 
                  test_dataset=test_dataset, 
                  optimizer=optimizer, 
                  scheduler=scheduler, 
                  model=model, 
                  loss_function=loss_func, 
                  logger=logger, 
                  warmup=warmup,
                  config=config,
                  resume=resume)