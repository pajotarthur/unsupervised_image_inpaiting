from .classify import MNISTExperiment


def get_experiment_by_name(name):
	if name == 'mnist':
		return MNISTExperiment
	raise NotImplementedError(name)

def init_experiment(name, **kwargs):
	args = kwargs.pop('args')
	return get_experiment_by_name(name)(**args, **kwargs)