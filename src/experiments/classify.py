import torch.optim as optim

from .base import EpochExperiment


class MNISTExperiment(EpochExperiment):

	def __init__(self, modules, datasets, lr, momentum, nepochs, device, niter, verbose):
		super(MNISTTrainer, self).__init__(modules, datasets)
		self.model = modules['model']
		self.train = datasets.get('train', [])
		self.val = datasets.get('val', [])
		self.test = datasets.get('test', [])
		self.lr = lr
		self.nepochs = nepochs
		self.device = device
		self.niter = niter
		self.verbose = verbose

		self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
		self.to(device)

	def __call__(self, batch, mode='train+eval'):
		self.train('train' in mode)

		batch = batch.to(self.device)
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