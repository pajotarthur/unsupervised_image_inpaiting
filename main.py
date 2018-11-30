
import torch
from tqdm import tqdm
from sacred import Experiment

from src.trainers import init_trainer
from src.datasets import init_dataset
from src.modules import init_module
from src.utils import init_config



def train(_run, config):

	# initializing datasets
	datasets = {}
	for dataset_name, dataset_config in config['datasets'].items():
		datasets[dataset_name] = init_dataset(**dataset_config)

	# initializing modules
	modules = {}
	for module_name, module_config in config['modules'].items():
		modules[module_name] = init_module(**module_config)

	# initializing trainer
	trainer = init_trainer(modules, datasets, **config['trainer'])

	epochs = tqdm(range(1, config['nepochs'] + 1), ncols=0)
	for epoch in epochs:
		trainer.step()
		epochs.set_description_str('Id: {} Epoch:{}/{} {}'.
			format(_run._id, epoch, config['nepochs'], trainer))


if __name__ == '__main__':

	config = init_config()
	ex = Experiment(config['exp_name'])
	ex.add_config({'config': config})
	ex.main(main)
	# ex.run('main')#config_updates={'config':config})
	# ex.run_commandline()
	# import ipdb; ipdb.set_trace() # BREAKPOINT


