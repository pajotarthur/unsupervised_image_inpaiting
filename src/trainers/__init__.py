from .classifier import ClassifierTrainer


def get_trainer_by_name(name):
	if name == 'classifier':
		return ClassifierTrainer
	raise NotImplementedError(name)

def init_trainer(modules, datasets, name, args):
	return get_trainer_by_name(name)(modules, datasets, args)