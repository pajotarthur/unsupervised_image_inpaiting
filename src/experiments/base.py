from collections import OrderedDict
from tqdm import tqdm, trange
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def size(batch):
    if isinstance(batch, dict):
        return batch[next(iter(batch))].shape[0]
    elif isinstance(batch, tuple) or \
        isinstance(batch, list):
        return batch[0].shape[0]
    else:
        raise NotImplementedError


class BaseExperiment(object):
    def __init__(self, device='cuda:0', verbose=1, train=True, evaluate=True):
        self.device = device
        self.verbose = verbose
        self.train = train
        self.evaluate = evaluate

    def update_state(self):
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

    def optimizers(self):
        for name, optimizer in self.named_optimizers():
            yield optimizer

    def named_optimizers(self):
        for name, optimizer in self._optimizers.items():
            yield name, optimizer

    def __setattr__(self, name, value):
        if isinstance(value, Module):
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
    
    def __delattr__(self, name):
        if name in self._modules:
            del self._modules[name]
        elif name in self._datasets:
            del self._datasets[name]
        elif name in self._optimizers:
            del self._optimizers[name]
        else:
            object.__delattr__(self, name)


class EpochExperiment(BaseExperiment):
    def __init__(self, nepochs=100, use_tqdm=True, niter='max', **kwargs):
        super(EpochExperiment, self).__init__(**kwargs)
        self.nepochs = nepochs
        self.use_tqdm = use_tqdm
        self.niter = niter

    def run(self, _run=None):
        self.metrics = self.init_metrics(_run)
        self.to_device()
        range = trange if self.use_tqdm else range
        for epoch in range(1, self.nepochs + 1):
            self.run_epoch(epoch, self.train, self.evaluate, _run)
            print(epoch)
            self.metrics.state.update(**self.update_state())
            self.metrics.reset()

    def run_epoch(self, epoch, train=True, evaluate=True, _run=None):
        for split, dataset in self.named_datasets():
            dataset = tqdm(dataset) if self.use_tqdm else dataset
            with torch.set_grad_enabled(train and (split == 'trainset')):
                metrics = getattr(self.metrics, split)
                for batch in dataset:
                    if isinstance(batch, (tuple, list)):
                        for i, v in enumerate(batch):
                            batch[i] = v.to(self.device)
                        output = self(*batch, train=(split=='trainset'), evaluate=evaluate)
                    elif isinstance(batch, dict):
                        for k, v in batch.items():
                            batch[k] = v.to(self.device)
                        output = self(**batch, train=(split=='trainset'), evaluate=evaluate)
                    else:
                        raise Error('Unknown batch type {}'.format(type(batch)))
                    metrics.update(**output, n=size(batch))
                    if self.use_tqdm:
                        dataset.set_postfix_str(str(metrics))

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