from torch.utils.data import DataLoader
import torch.nn as nn

from collections import OrderedDict


def size(batch):
	return next(iter(batch)).shape[0]



class BaseExperiment(object):
	def run(self, mode='train+eval'):
		for epoch in range(1, self.nepochs + 1):
			self.run_epoch(epoch)
			print(self)
	
	def run_epoch(self, epoch):
		self.metrics.reset()
		
		self.metrics['state'](**self.update_state(epoch))

		for batch in self.train:
			self.logger.Parent_Train()
			self.metrics['train'](**self(batch, mode='train+eval'), n=size(batch))

		for batch in self.val:
			self.metrics['val'](**self(batch, mode='eval'), n=size(batch))

		for batch in self.test:
			self.metrics['test'](**self(batch, mode='eval'), n=size(batch))
	
	def update_state(self, epoch):
		return self.get_state()

	def get_state(self):
		return

	def train(self, mode=True):
		torch.no_grad(not mode)
		for m in self.modules():
			m.train(mode)
	
	def eval(self):
		self.train(mode=False)

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
	# 	if isinstance()
	# 	raise NotImplementedError



class EpochExperiment(BaseExperiment):
	def run(self, mode='train+eval'):
		for epoch in range(1, self.nepochs + 1):
			self.run_epoch(epoch)
			print(self)
	
	def run_epoch(self, epoch):
		self.metrics.reset()
		self.metrics['state'](**self.update_state(epoch))

		for batch in self.train:
			self.logger.Parent_Train()
			self.metrics['train'](**self(batch, mode='train+eval'), n=size(batch))

		for batch in self.val:
			self.metrics['val'](**self(batch, mode='eval'), n=size(batch))

		for batch in self.test:
			self.metrics['test'](**self(batch, mode='eval'), n=size(batch))

	def __str__(self):
		s = ''
		if self.verbose > 0:
			s += ' '.join([m.__str__() for m in self.metrics['eval']])
		if self.verbose > 1:
			s += ' '.join([m.__str__() for m in self.metrics['train']])
		if self.verbose > 2:
			s += ' '.join([m.__str__() for m in self.metrics['state']])
		return s