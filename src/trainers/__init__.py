from .classifier import ClassifierTrainer


def get_trainer_by_name(name):
	if name == 'classifier':
		return ClassifierTrainer
	else:
		raise NotImplementedError(name)

def init_trainer(modules, config):
	name, args = config['name'], config['args']
	return get_trainer_by_name(name)(modules, args)