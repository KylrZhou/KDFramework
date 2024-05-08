from utils import LOGGER

import os
import json
import time
import wandb
import yaml
import torch
from copy import deepcopy

monthlist = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

@LOGGER.register()
class Logger():
    def __init__(self,
                 log_interval, 
                 checkpoint_interval,
                 MAX_EPOCH,
                 config,
                 experiment_name='', 
                 float_tolerance=4, 
                 SavePath='/home/usr00/KDFrameworkDATA/logs',
                 CalcEpochAvgValue=False,
                 Print2Terminal=True,
                 Write2File=False,
                 SaveCheckpoint=False,
                 Upload2Wandb=False,
                 MainScoreName="Acc",
                 TimeLog = True):
        self.SavePath = SavePath
        self.EPOCH = 1
        self.ITER = 0
        self.float_tolerance = float_tolerance
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.MAX_EPOCH = int(MAX_EPOCH)
        self.interval_counter = 0
        self.Ibuffer = {}
        self.Ebuffer = {}
        self.Vbuffer = {}
        self.Ebuffer_counter = 0
        self.Stamp = ''
        self.LocalTime = self.get_ltime()

        self.CalcEpochAvgValue = CalcEpochAvgValue
        self.Print2Terminal = Print2Terminal
        self.Write2File = Write2File
        self.SaveCheckpoint = SaveCheckpoint
        self.Upload2Wandb = Upload2Wandb
        self.MainScoreName = MainScoreName

        # not yet realized
        self.TimeLog = TimeLog
        self.data_timer = 0
        self.calc_timer = 0
        self.epoch_time = 0
        self.avg_epoch_time = 0
        self.BestScore = -1
        if self.Write2File or self.SaveCheckpoint:
            self.make_path(config_name=config['filename'], config=config)
        if self.Upload2Wandb:
            wandb.login()
            wandb.init(project=config['settings']['project'], 
                       name=config['filename']+'_'+self.LocalTime,
                       dir=config['settings']['wandb_dir'])
    
    def log(self, Value, Tag):
        if isinstance(Value, torch.Tensor):
            _Value = Value.item()
        else:
            _Value = Value
        _Value = self.float_round(_Value, self.float_tolerance)
        try:
            self.Ibuffer[Tag] += _Value
        except KeyError:
            self.new_entry(self.Ibuffer, Tag)
            self.Ibuffer[Tag] += _Value
        try:
            self.Ebuffer[Tag] += _Value
        except KeyError:
            self.new_entry(self.Ebuffer, Tag)
            self.Ebuffer[Tag] += _Value
        return Value
    
    def log_val(self, Value, Tag):
        if isinstance(Value, torch.Tensor):
            Value = Value.item()
        self.Vbuffer[Tag] = self.float_round(Value, self.float_tolerance)
        return Value
    
    def update(self):
        self.ITER += 1
        self.interval_counter += 1
        iter_log = self.attach_stamp(self.Ibuffer, 'ITER')
        # write every iter to file
        self.write_to_file(iter_log)
        # upload every iter to wandb
        self.upload_to_wandb(iter_log)
        # initialize iter buffer
        self.init_buffer(self.Ibuffer)
        # if iter is not log iter and final iter, do nothing
        if self.interval_counter != self.log_interval\
            and self.ITER != self.MAX_ITER:
            return
        # printing log to terminal
        self.printing_to_terminal(iter_log)
        # reset interval counter to 1
        self.interval_counter = 0
        # if iter is final one
        if self.ITER >= self.MAX_ITER:
            if self.CalcEpochAvgValue:
                # average the epoch buffer
                epoch_log = self.epoch_avg()
                epoch_log = self.attach_stamp(epoch_log, 'EPOCH')
                # printing log to terminal
                self.printing_to_terminal(epoch_log)
                # write to file
                self.write_to_file(epoch_log)
                # upload to wandb
                self.upload_to_wandb(epoch_log)
                # initialize epoch buffer
            self.init_buffer(self.Ebuffer)
            self.EPOCH += 1
            self.ITER = 0
    
    def init_model_optimizer_scheduler(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def init_dataset(self, dataset):
        self.MAX_ITER = len(dataset)
    
    def update_val(self):
        log_val = self.attach_stamp(self.Vbuffer, 'VAL')
        self.printing_to_terminal(log_val)
        self.write_to_file(log_val)
        self.upload_to_wandb(log_val)
        if log_val[self.MainScoreName] >= self.BestScore:
            p = deepcopy(self.BestScore)
            self.BestScore = log_val[self.MainScoreName]
            self.log_checkpoint('BEST', p = p)
        if (self.EPOCH-1) % self.checkpoint_interval == 0 and self.EPOCH != 1:
            self.log_checkpoint('PERIODIC')
    
    def log_checkpoint(self, status, p=None):
        if self.SaveCheckpoint == False:
            return
        if status == 'BEST':
            if self.Print2Terminal:
                print(f"EPOCH:{self.EPOCH-1}\tBest Score Updated: {p} -> {self.BestScore}")
            ckpt = {'EPOCH':self.EPOCH-1,
                    'BESTSCORE':self.BestScore,
                    'MODEL':self.model.state_dict(),
                    'OPTIMIZER':self.optimizer.state_dict(),
                    'SCHEDULER':self.scheduler.state_dict()}
            torch.save(ckpt, os.path.join(self.SavePath, 'checkpoints', "Best.pth"))
        elif status == 'PERIODIC':
            ckpt = {'EPOCH':self.EPOCH-1,
                    'BESTSCORE':self.BestScore,
                    'MODEL':self.model.state_dict(),
                    'OPTIMIZER':self.optimizer.state_dict(),
                    'SCHEDULER':self.scheduler.state_dict()}
            torch.save(ckpt, os.path.join(self.SavePath, 'checkpoints', f"Epoch_{self.EPOCH-1}.pth"))

    def init_buffer(self, buffer):
        for k in buffer.keys():
            buffer[k] = 0
    
    def new_entry(self, buffer, Tag):
        buffer[Tag] = 0

    def attach_stamp(self, tar, buffer_type):
        temp = {}
        temp['Epoch'] = self.EPOCH
        if buffer_type == 'ITER':
            self.epoch_time += self.calctime
            self.epoch_time += self.datatime 
            tar.update({"CalcTime":self.calctime,"DataTime":self.datatime})
            temp['Iter'] = self.ITER
        elif buffer_type == 'VAL':
            self.epoch_time += self.calctime
            tar.update({"ValTime":self.calctime, "EpochTime":f"{int(self.epoch_time/60)}m{int(self.epoch_time%60)}s", "eta":self.calc_eta_time()})
            temp['Epoch'] -= 1
        temp.update(tar)
        return temp

    def write_to_file(self, buffer):
        if self.Write2File == False:
            return
        # convert temp into json
        temp = json.dumps(buffer)
        self.logpath.write(f"{temp}\n")

    def upload_to_wandb(self, buffer):
        if self.Upload2Wandb == False:
            return
        wandb.log(buffer)

    def printing_to_terminal(self, buffer):
        if self.Print2Terminal == False:
            return
        temp = ''
        _buffer = deepcopy(buffer)
        try:
            _ = _buffer['Iter']
            _buffer['Iter'] = f"[{_}/{self.MAX_ITER}]"
        except KeyError:
            pass
        for k, v in _buffer.items():
            if k != 'Epoch' and isinstance(v, float):
                temp += f'{k}: {v:.4f}\t'
            else:
                temp += f"{k}:{v}\t"
        print(temp)

    def epoch_avg(self):
        temp = {}
        for k, v in self.Ebuffer.items():
            temp[f"eAvg{k}"] = self.float_round(v/self.ITER, self.float_tolerance+1)
            self.Ebuffer[k] = 0
        return temp
    
    def data_time_start(self):
        self.data_timer = time.time()

    def data_time_end(self):
        self.datatime = self.float_round(time.time() - self.data_timer, self.float_tolerance)

    def calc_time_start(self):
        self.calc_timer = time.time()

    def calc_time_end(self):
        self.calctime = self.float_round(time.time() - self.calc_timer, self.float_tolerance)
        
    def float_round(self, value, tolerance):
        if isinstance(value, float):
            _ = float(10**tolerance)
            value = int(value * _ + 0.5)/_
        return value
    
    def get_ltime(self):
        ltime = time.localtime()
        return f"{ltime[0]-2000}{monthlist[ltime[1]-1]}{ltime[2]:02d}_{ltime[3]:02d}{ltime[4]:02d}{ltime[5]:02d}"

    def make_path(self, config_name, config):
        self.SavePath = os.path.join(self.SavePath, config_name)
        if os.path.isdir(self.SavePath) == False:
            os.makedirs(self.SavePath)
        self.SavePath = os.path.join(self.SavePath, self.LocalTime)
        os.mkdir(self.SavePath)
        if self.SaveCheckpoint:
            os.mkdir(os.path.join(self.SavePath, 'checkpoints'))
        if self.Write2File:
            self.logpath = os.path.join(self.SavePath, 'log.json')
            self.logpath = open(self.logpath, 'a')
            with open(os.path.join(self.SavePath, 'config.yaml'), 'w') as f:
                f.write(yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False))

    def calc_eta_time(self):
        self.avg_epoch_time += ((self.epoch_time/60)-self.avg_epoch_time)/(self.EPOCH-1)
        tmp = self.avg_epoch_time * (self.MAX_EPOCH - self.EPOCH - 1)
        h = int(tmp/60)
        m = int(tmp%60+1)
        self.epoch_time = 0
        return f"{h}h{m}m"