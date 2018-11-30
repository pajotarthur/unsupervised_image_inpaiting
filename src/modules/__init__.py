from functools import partial

from .classifiers import MnistClassifier


def get_module_by_name(name):
	if name == 'mnist_classifier':
		return MnistClassifier
	# elif:
	# 	return partial(init_generator, SaganGenerator)
	raise NotImplementedError(name)

def init_module(name, args):
	return get_module_by_name(name)(**args)
