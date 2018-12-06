from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

def size(batch):
    if isinstance(batch, dict):
        return batch[next(iter(batch))].shape[0]
    elif isinstance(batch, tuple) or \
        isinstance(batch, list):
        return batch[0].shape[0]
    else:
        raise NotImplementedError


class BaseExperiment(object):
    def run(self, mode='train+eval'):
        for epoch in range(1, self.nepochs + 1):
            self.run_epoch(epoch)
            print(self)
    
    def update_state(self, epoch):
        return self.get_state()

    def get_state(self):
        return {}

    def train_mode(self, mode=True):
        for m in self.modules():
            m.train(mode)
    
    def eval_mode(self):
        self.train_mode(mode=False)

    def to(self, device):
        for m in self.modules():
            m.to(device)

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def named_modules(self):
        for name, module in self._modules.items():
            yield name, module

    def datasets(self):
        for name, dataset in self.named_datasets():
            yield dataset

    def named_datasets(self):
        for name, dataset in self._datasets.items():
            yield name, dataset

    def __setattr__(self, name, value):
        if isinstance(value, nn.Module):
            if not hasattr(self, '_modules'):
                self._modules = OrderedDict()
            self._modules[name] = value
        elif isinstance(value, DataLoader):
            if not hasattr(self, '_datasets'):
                self._datasets = OrderedDict()
            self._datasets[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_datasets' in self.__dict__:
            datasets = self.__dict__['_datasets']
            if name in datasets:
                return datasets[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))     
    
    # def __delattr__(self, name):
    #   if isinstance()
    #   raise NotImplementedError


class EpochExperiment(BaseExperiment):
    def run(self, mode='train+eval'):
        epochs = trange(1, self.nepochs + 1) if self.use_tqdm else range(1, self.nepochs)
        for epoch in epochs:
            self.run_epoch(epoch)
    
    def run_epoch(self, epoch):
        self.metrics.state.update(**self.update_state(epoch))

        train = tqdm(self.train) if self.use_tqdm else self.train
        with torch.set_grad_enabled(True):
            for batch in train:
                self.metrics.train.update(**self(**batch, mode='train+eval'), n=size(batch))
                if self.use_tqdm:
                    train.set_postfix_str(str(self.metrics.train))

        test = tqdm(self.test) if self.use_tqdm else self.test
        with torch.set_grad_enabled(False):
            for batch in test:
                self.metrics.test.update(**self(**batch, mode='eval'), n=size(batch))
                if self.use_tqdm:
                    test.set_postfix_str(str(self.metrics.test))

        self.metrics.reset()

    def __str__(self):
        return str(self.metrics)
        s = ''
        if self.verbose > 0:
            s += str(self.metrics.test)
        if self.verbose > 1:
            s += str(self.metrics.train)
        if self.verbose > 2:
            s += str(self.metrics.state)
        return s