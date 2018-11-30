import src.utils.meters as meters


class BaseTrainer():
	def __init__(self, modules, datasets, **kwargs):
		self._modules = modules
		self._datasets = datasets
		self._meters = {}

	def train(self):
		for name, dataset in self.datasets.items():
			iter = 0
			self.reset_metrics()
			for batch in dataset:
				iter += 1
				self.forward(batch)
				bs = len(batch[list(batch.keys())[0]])
				self.update_metrics(
					batch_size=bs
				)

				if (niter[name] != 'max') and (iter > niter):
					break
	
	def forward(self):
		raise NotImplementedError

	def eval(self):
		raise NotImplementedError

	def register_metric(self, name, type='scalar'):
		if type == 'scalar':
			self._meters[name] = meters.ScalarMeter()
		elif type == 'average':
			self._meters[name] = meters.AverageMeter()
		else:
			raise NotImplementedError(name)

	def update_meters(self, n=1):
		for name, meter in self._meters.item():
			if not hasattr(self, name):
				raise Error('You have to log {}'.format(name))
			value = getattr(self, name)
			meter.update(value, n=n)

	def __call__(self, *args, **kwargs):
		for meter in self._meters.values():
			meter.reset()
		self.forward(*args, **kwargs)
		return self._meters

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
	
	def __delattr__(self, name):
		if name in self._modules:
			del self._modules[name]
		elif name in self._datasets:
			del self._datasets[name]
		else:
			object.__delattr__(self, name)

	def __repr__(self):
		s = ''
		for name, metric in self._meters.items():
			s += '{}'.format(metric.__repr__)
		return s
