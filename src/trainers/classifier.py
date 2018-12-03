from .base import BaseTrainer
import torch.optim as optim


class MNISTTrainer(BaseTrainer):

	def __init__(self, modules, datasets, lr, momentum, nepochs, device, niter):
		super(MNISTTrainer, self).__init__(modules, datasets)
		self.lr = lr
		self.nepochs = nepochs
		self.device = device
		self.niter = niter

		self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
		self.model.to(device)

	def train(self, mode=True):
		torch.no_grad(not mode)
		for m in self._modules.values():
			m.train(mode)

	def eval(self):
		self.train(mode=False)

	def get_state(self):
		return 0

	def run(self, mode='train+eval'):
		for epoch in range(1, self.nepochs + 1):
			print('epoch', epoch)
			self.meters['state'].update(self.get_state())
			self._train_epoch()

	def _train_epoch(self):

		for batch in self.train:
			self.meters['train'](self(batch, mode='train+eval'))

		for batch in self.val:
			self.meters['val'](self(batch, mode='eval'))

	def __call__(self, batch, mode='train+eval'):
		self.train('train' in mode)

		batch = batch.to(self.device)
		output = self.model(batch['sample'])
		loss = F.nll_loss(output, batch['class'])

		if 'train' in mode:
			self.optim.zero_grad()
			self.loss.backward()
			self.optim.step()

		if 'eval' in mode:
			self.eval()
			pred = output.max(1, keepdim=True)[1]
			correct = pred.eq(target.view_as(pred)).sum()

		return {
			'loss': loss,
			'percent': correct,
		}