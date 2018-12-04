import torch.optim as optim

from .base import EpochExperiment


class MNISTExperiment(EpochExperiment):

	def __init__(self, model, train=[], val=[], test=[],
		lr=1e-3, momentum=0.9, nepochs=10, device='cuda:0',
		niter='max', verbose=1):

		self.model = model
		self.train = train
		self.val = val
		self.test = test
		self.lr = lr
		self.nepochs = nepochs
		self.device = device
		self.niter = niter
		self.verbose = verbose

		self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
		self.to(device)

	def __call__(self, batch, mode='train+eval'):
		self.train('train' in mode)

		for b in batch.values():
			b.to(self.device)
		output = self.model(batch['sample'])
		loss = F.nll_loss(output, batch['class'])

		if 'train' in mode:
			self.optim.zero_grad()
			self.loss.backward()
			self.optim.step()

		eval_results = {}
		if 'eval' in mode:
			self.eval()
			pred = output.max(1, keepdim=True)[1]
			correct = pred.eq(target.view_as(pred)).sum()
			eval_results['correct'] = correct

		return {
			'loss': loss,
			**eval_results,
		}