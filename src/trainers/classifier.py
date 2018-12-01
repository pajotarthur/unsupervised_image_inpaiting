from .base import BaseTrainer


class ClassifierTrainer(BaseTrainer):
	def __init__(self, modules, datasets, niter, eval_on_train=True,):
		super(ClassifierTrainer, self).__init__(modules, datasets)
		self.eval_on_train = eval_on_train
		self.niter = niter if niter == 'max' else niter
		self.register_metric('loss_train', type='scalar')
		self.register_metric('loss_val', type='scalar')
		# self.register_logger() pourquoi pas généraliser metric, aux images, etc?
		
	# def forward(self):
	# 	self.loss_train = 
	# 	self.loss_val = 

	def __repr__(self):
		return str(10)
			

