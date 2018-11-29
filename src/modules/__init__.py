from .classifiers import MnistClassifier


def get_module_by_name(name):
	if name == 'mnist_classifier':
		return MnistClassifier
	else:
		raise NotImplementedError(name)

def init_module(config):
	name, args = config['name'], config['args']
	return get_module_by_name(name)(**args)
