from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm, trange


def size(batch):
    if isinstance(batch, dict):
        return batch[next(iter(batch))].shape[0]
    elif isinstance(batch, tuple) or \
        isinstance(batch, list):
        return batch[0].shape[0]
    else:
        raise NotImplementedError

def to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)


class BaseExperiment(object):
    def __init__(self, device='cuda:0', verbose=1):
        self.device = device
        self.verbose = verbose

    def update_state(self, epoch):
        return self.get_state()

    def get_state(self):
        return {}

    def train_mode(self, mode=True):
        for m in self.modules():
            m.train(mode)
    
    def eval_mode(self):
        self.train_mode(mode=False)

    def to_device(self):
        for m in self.modules():
            m.to(self.device)

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
        elif isinstance(value, Optimizer):
            if not hasattr(self, '_optimizers'):
                self._optimizers = OrderedDict()
            self._optimizers[name] = value
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
        if '_optimizers' in self.__dict__:
            optimizers = self.__dict__['_optimizers']
            return optimizers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))     
    
    # def __delattr__(self, name):
    #   if isinstance()
    #   raise NotImplementedError


class EpochExperiment(BaseExperiment):
    def __init__(self, nepochs=100, use_tqdm=True, niter='max', mode='train+eval', **kwargs):
        super(EpochExperiment, self).__init__(**kwargs)
        self.nepochs = nepochs
        self.use_tqdm = use_tqdm
        self.niter = niter
        self.mode = mode

    def run(self, _run=None, mode='train+eval'):
        self.metrics = self.init_metrics(_run)
        self.to_device()
        epochs = trange(1, self.nepochs + 1) if self.use_tqdm else range(1, self.nepochs)
        for epoch in epochs:
            self.run_epoch(epoch, self.mode, _run)
            self.metrics.reset()

    def run_epoch(self, epoch, mode, _run=None):
        self.metrics.state.update(**self.update_state(epoch))

        # attempt at making it generic
        # for split, dataset in self.named_datasets():
        #     dataset = tqdm(dataset) if self.use_tqdm else dataset
        #     with torch.set_grad_enabled(('train' in mode) and (split == 'train')):
        #         for batch in dataset:
        #             to_device(batch, self.device)
        #             mode = 'train+eval' if split=='train' else 'eval'
        #             output = self(**batch, mode=mode)
        #             self.metrics.dataset.update(**output, n=size(batch))
        #             if self.use_tqdm:
        #                 dataset.set_postfix_str(str(self.metrics.dataset))

        train = tqdm(self.train) if self.use_tqdm else self.train
        with torch.set_grad_enabled(True):
            for batch in train:
                to_device(batch, self.device)
                output = self(**batch, mode='train+eval')
                self.metrics.train.update(**output, n=size(batch))
                if self.use_tqdm:
                    s = str(getattr(self.metrics, split))
                    train.set_postfix_str(s)

        test = tqdm(self.test) if self.use_tqdm else self.test
        with torch.set_grad_enabled(False):
            for batch in test:
                to_device(batch, self.device)
                output = self(**batch, mode='eval')
                self.metrics.test.update(**output, n=size(batch))
                if self.use_tqdm:
                    test.set_postfix_str(str(self.metrics.test))

    def __str__(self):
        return str(self.metrics)
        # s = ''
        # if self.verbose > 0:
        #     s += str(self.metrics.test)
        # if self.verbose > 1:
        #     s += str(self.metrics.train)
        # if self.verbose > 2:
        #     s += str(self.metrics.state)
        # return s