from .classifier import MNISTExperiment


def get_experiment_by_name(name):
	if name == 'mnist':
		return MNISTExperiment
	raise NotImplementedError(name)

def init_experiment(modules, datasets, name, args):
	return get_experiment_by_name(name)(modules, datasets, **args)