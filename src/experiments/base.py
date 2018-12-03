import src.utils.meters as meters


def size(batch):
	return next(iter(batch)).shape[0]


class EpochExperiment(object):
	def __init__(self, modules, datasets, **kwargs):
		self._modules = modules
		self._datasets = datasets
		self._meters = {}

	def run(self, mode='train+eval'):
		for epoch in range(1, self.nepochs + 1):
			self.run_epoch(epoch)
			print(self)
	
	def run_epoch(self, epoch):
		self.meters['state'](**self.update_state(epoch))

		for batch in self.train:
			self.meters['train'](**self(batch, mode='train+eval'), n=size(batch))

		for batch in self.val:
			self.meters['val'](**self(batch, mode='eval'), n=size(batch))

		for batch in self.test:
			self.meters['test'](**self(batch, mode='eval'), n=size(batch))
	
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

	def __repr__(self):
		s = ''
		if self.verbose > 0:
			s += ' '.join([m.__repr__() for m in self.meters['eval']])
		if self.verbose > 1:
			s += ' '.join([m.__repr__() for m in self.meters['train']])
		if self.verbose > 2:
			s += ' '.join([m.__repr__() for m in self.meters['state']])
		return s