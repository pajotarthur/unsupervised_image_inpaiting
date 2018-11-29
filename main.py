
import torch

from src.trainers import init_trainer
from src.datasets import init_dataset
from src.modules import init_module

from sacred import Experiment



ex = Experiment('test')

@ex.main
def train(datasets, modules, trainer):

	# initializing datasets
    datasets = {}
    for dataset_name, dataset_config in config['datasets'].items():
        datasets[dataset_name] = init_dataset(dataset_config)

	# initializing modules
    modules = {}
    for module_name, module_config in config['modules'].items():
        modules[module_name] = init_module(module_config)

	# initializing trainer
    trainer = init_trainer(modules, config['trainer'])

	# epochs = tqdm(range(1, config['nepochs'] + 1), ncols=0)
	# for epoch in epochs:
	# 	logs = trainer.train(dataset['train'])
	# 	logs.update(trainer.eval(dataset['train']))
	# 	logs.update(trainer.eval(dataset['test']))


if __name__ == '__main__':
	from sacred.config import load_config_file
	config = load_config_file('config.yaml')

	modules_default_configs = load_config_file('config/modules.yaml')
	for module_name, module_config in config['modules'].items():
		name, args = module_config['name'], module_config['args']
		new_args = modules_default_configs[name]
		new_args.update(args)
		module_config['args'] = new_args

	datasets_default_configs = load_config_file('config/datasets.yaml')
	for dataset_name, dataset_config in config['datasets'].items():
		name, args = dataset_config['name'], dataset_config['args']
		new_args = datasets_default_configs[name]
		new_args.update(args)
		dataset_config['args'] = new_args

	trainer_default_configs = load_config_file('config/trainer.yaml')
	name, args = config['trainer']['name'], config['trainer']['args']
	new_args = trainer_default_configs[name]
	new_args.update(args)
	config['trainer']['args'] = new_args

	ex.add_config(config)
	ex.run_commandline()
	# config = {
	# 	'modules':{},
	# 	'datasets':{},
	# 	'trainers':{},
	# }
	# ex.main(train)
	# ex.run(config_updates={'config':config})

	# import ipdb; ipdb.set_trace() # BREAKPOINT


