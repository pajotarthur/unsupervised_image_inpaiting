from .classifier import MNISTTrainer


def get_trainer_by_name(name):
	if name == 'mnist':
		return MNISTTrainer
	raise NotImplementedError(name)

def init_trainer(modules, datasets, name, args):
	return get_trainer_by_name(name)(modules, datasets, **args)