import os
import json
import time
import torch
from copy import deepcopy

monthlist = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

class Logger():
    def __init__(self,
                 log_interval, 
                 checkpoint_interval,
                 MAX_ITER,
                 experiment_name='', 
                 float_tolerance=4, 
                 LocalPath='./logs',
                 CalcEpochAvgValue=False,
                 Print2Terminal=True,
                 Write2File=False,
                 Upload2Wandb=False,
                 MainScoreName="Acc",
                 TimeLog = True):
        self.LocalPath = LocalPath
        self.EPOCH = 1
        self.ITER = 0
        self.MAX_ITER = MAX_ITER
        self.float_tolerance = float_tolerance
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.interval_counter = 0
        self.Ibuffer = {}
        self.Ebuffer = {}
        self.Vbuffer = {}
        self.Ebuffer_counter = 0
        self.Stamp = ''
        self.LocalTime = ''
        self.CalcEpochAvgValue = CalcEpochAvgValue
        self.Print2Terminal = Print2Terminal
        self.Write2File = Write2File
        self.Upload2Wandb = Upload2Wandb
        self.MainScoreName = MainScoreName
        self.TimeLog = TimeLog
        self.data_timer = 0
        self.calc_timer = 0
        self.BestScore = -1
        
        self.make_path(experiment_name)
    
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
        ###self.upload_to_wandb(iter_log)
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
                ###self.upload_to_wandb(epoch_log)
                # initialize epoch buffer
            self.init_buffer(self.Ebuffer)
            self.EPOCH += 1
            self.ITER = 0
    
    def init_model(self, model):
        self.model = model
    
    def update_val(self):
        log_val = self.attach_stamp(self.Vbuffer, 'VAL')
        self.printing_to_terminal(log_val)
        self.write_to_file(log_val)
        if log_val[self.MainScoreName] >= self.BestScore:
            self.BestScore = log_val[self.MainScoreName]
            self.log_checkpoint('BEST')
        if (self.EPOCH-1) % self.checkpoint_interval == 0 and self.EPOCH != 1:
            self.log_checkpoint('PERIODIC')
    
    def log_checkpoint(self, status):
        if status == 'BEST':
            torch.save(self.model, os.path.join(self.LocalPath, 'checkpoints', "Best.pth"))
        elif status == 'PERIODIC':
            torch.save(self.model, os.path.join(self.LocalPath, 'checkpoints', f"Epoch_{self.EPOCH-1}.pth"))
        

    def init_buffer(self, buffer):
        for k in buffer.keys():
            buffer[k] = 0
    
    def new_entry(self, buffer, Tag):
        buffer[Tag] = 0

    def attach_stamp(self, tar, buffer_type):
        temp = {}
        temp['Epoch'] = self.EPOCH
        if buffer_type == 'ITER':
            tar.update({"CalcTime":self.calctime,"DataTime":self.datatime})
            temp['Iter'] = self.ITER
        elif buffer_type == 'VAL':
            tar.update({"ValTime":self.calctime})
            temp['Epoch'] -= 1
        temp.update(tar)
        return temp

    def write_to_file(self, buffer):
        if self.write_to_file:
            # convert temp into json
            temp = json.dumps(buffer)
            self.logpath.write(f"{temp}\n")

    def upload_to_wandb(self, buffer):
        pass
        #wandb.log(temp)

    def printing_to_terminal(self, buffer):
        if self.Print2Terminal:
            temp = ''
            _buffer = deepcopy(buffer)
            try:
                _ = _buffer['Iter']
                _buffer['Iter'] = f"[{_}/{self.MAX_ITER}]"
            except KeyError:
                pass
            for k, v in _buffer.items():
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
    
    def make_path(self, name=''):
        if os.path.isdir(self.LocalPath) == False:
            os.mkdir(self.LocalPath)
        ltime = time.localtime()
        ltime = f"{ltime[0]-2000}{monthlist[ltime[1]-1]}{ltime[2]:02d}_{ltime[3]:02d}{ltime[4]:02d}{ltime[5]:02d}"
        name = ltime+name
        self.LocalPath = os.path.join(self.LocalPath, name)
        os.mkdir(self.LocalPath)
        os.mkdir(os.path.join(self.LocalPath, 'checkpoints'))
        self.logpath = os.path.join(self.LocalPath, 'log.json')
        self.logpath = open(self.logpath, 'a')