import yaml
from torchvision.transforms import Compose

from utils import DATASETTRAIN
from utils import DATASETVAL
from utils import MODEL
from utils import PIPELINE
from utils import AUGMENTATION
from utils import OPTIMIZER
from utils import SCHEDULER
from utils import LOGGER
from utils import FUNCTION
from utils import DISTILLER

def build_model(opt):
    TYPE = opt['TYPE']
    opt.pop('TYPE')
    if isinstance(opt, dict):
        model = MODEL.get(TYPE)(**opt)
    elif isinstance(opt, str):
        model = MODEL.get(TYPE)()
    elif isinstance(opt, list):
        model = MODEL.get(TYPE)(*opt)
    return model

def build_pipeline(opt):
    TYPE = opt['TYPE']
    opt.pop('TYPE')
    if isinstance(opt, dict):
        pipeline = PIPELINE.get(TYPE)(**opt)
    elif isinstance(opt, str):
        pipeline = PIPELINE.get(TYPE)()
    elif isinstance(opt, list):
        pipeline = PIPLINE.get(TYPE)(*opt)
    return pipeline

def build_dataset_train(opt):
    TYPE = opt['TYPE']
    if TYPE[-6:] == '_train':
        pass
    else:
        TYPE = TYPE + '_train'
    opt.pop('TYPE')
    if isinstance(opt, dict):
        dataset = DATASETTRAIN.get(TYPE)(**opt)
    elif isinstance(opt, str):
        dataset = DATASETTRAIN.get(TYPE)()
    elif isinstance(opt, list):
        dataset = DATASETTRAIN.get(TYPE)(*opt)
    return dataset

def build_dataset_val(opt):
    TYPE = opt['TYPE']
    if TYPE[-4:] == '_val':
        pass
    else:
        TYPE = TYPE + '_val'
    opt.pop('TYPE')
    if isinstance(opt, dict):
        dataset = DATASETVAL.get(TYPE)(**opt)
    elif isinstance(opt, str):
        dataset = DATASETVAL.get(TYPE)()
    elif isinstance(opt, list):
        dataset = DATASETVAL.get(TYPE)(*opt)
    return dataset

def build_augmentation(aug, params):
    if isinstance(params, dict):
        augmentation = AUGMENTATION.get(aug)(**params)
    elif params is None:
        augmentation = AUGMENTATION.get(aug)()
    return augmentation

def build_augmentation_pipeline(opt):
    opt = yaml.load(open(opt, 'r'), Loader=yaml.FullLoader)
    pipeline = []
    for k, v in opt.items():
        pipeline.append(build_augmentation(k, v))
    pipeline = Compose(pipeline)
    return pipeline

def build_optimizer(model_params, opt):
    TYPE = opt['TYPE']
    opt.pop('TYPE')
    for k, v in opt.items():
        if isinstance(v, str):
            opt[k] = eval(v)
    if isinstance(opt, dict):
        optimizer = OPTIMIZER.get(TYPE)(model_params, **opt)
    elif isinstance(opt, str):
        optimizer = OPTIMIZER.get(TYPE)(model_params)
    elif isinstance(opt, list):
        optimizer = OPTIMIZER.get(TYPE)(model_params, *opt)
    return optimizer

def build_scheduler(optimizer, opt):
    TYPE = opt['TYPE']
    opt.pop('TYPE')
    if isinstance(opt, dict):
        scheduler = SCHEDULER.get(TYPE)(optimizer, **opt)
    elif isinstance(opt, str):
        scheduler = SCHEDULER.get(TYPE)(optimizer)
    elif isinstance(opt, list):
        scheduler = SCHEDULER.get(TYPE)(optimizer, *opt)
    return scheduler

def build_logger(opt):
    logger_config = opt['logger']
    logger_config['config'] = opt
    TYPE = logger_config['TYPE']
    logger_config.pop('TYPE')
    if isinstance(logger_config, dict):
        logger = LOGGER.get(TYPE)(**logger_config)
    elif isinstance(logger_config, str):
        logger = LOGGER.get(logger_config)()
    elif isinstance(logger_config, list):
        logger = LOGGER.get(TYPE)(*logger_config)
    return logger

def build_fucntion(opt):
    TYPE = opt['TYPE']
    opt.pop('TYPE')
    if isinstance(opt, dict):
        function = FUNCTION.get(TYPE)(**opt)
    elif isinstance(opt, str):
        function = FUNCTION.get(TYPE)()
    elif isinstance(opt, list):
        function = FUNCTION.get(TYPE)(*opt)
    return function


"""
def build_distiller(opt):
    TYPE = opt['TYPE']
    opt.pop('TYPE')
    if isinstance(opt['teacher'], dict):
        opt['teacher'] = build_model(opt['teacher'])
    elif isinstance(opt['teacher'], list):
        temp = []
        for i in opt['teacher']:
            temp.append(build_model(i))
        opt['teacher'] = temp
    if isinstance(opt['kd_loss_fucntion'], dict):
        opt['kd_loss_fucntion'] = build_fucntion(opt['kd_loss_fucntion'])
    elif isinstance(opt['kd_loss_fucntion'], list):
        temp = []
        for i in opt['kd_loss_fucntion']:
            temp.append(build_fucntion(i))
        opt['kd_loss_fucntion'] = temp
    if isinstance(opt, dict):
        distiller = DISTILLER.get(TYPE)(**opt)
    elif isinstance(opt, str):
        distiller = DISTILLER.get(TYPE)()
    elif isinstance(opt, list):
        distiller = DISTILLER.get(TYPE)(*opt)
    return distiller
"""
def build_distiller(opt):
    TYPE = opt['TYPE']
    opt.pop('TYPE')
    if isinstance(opt, dict):
        distiller = DISTILLER.get(TYPE)(**opt)
    elif isinstance(opt, str):
        distiller = DISTILLER.get(TYPE)()
    elif isinstance(opt, list):
        distiller = DISTILLER.get(TYPE)(*opt)
    return distiller