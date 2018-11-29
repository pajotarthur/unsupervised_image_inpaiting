
import torch
# import src
import yaml
from sacred import Experiment

def init_datasets(config):
	datasets = {}
	for name, v in config.items():
		datasets[name] = src.datasets[v['type']](**v['args'])
	return datasets

def init_modules(config):
	modules = {}
	for name, v in config.items():
		modules[name] = src.modules[v['type']](**v['args'])
	return modules

def init_trainer(modules, config):
	trainer_func = src.trainers[config['type']]
	return trainer_func(modules, **config['args'])

def save_logs(logs, ex):
	for k in logs:

		if isinstance(logs[k], (int, float)):
			ex.log_scalar(logs[k], )

	

ex = Experiment('test')

# default = {
# 	'ok': 1,
# 	'nok': 2,
# }

# @ex.config
# def config():
# 	modules, datasets, trainer = 12, {}, {}
# 	print(modules)
# 	modules = {'type':'ok'}
# 	print(modules)
	# print(modules)
	# modules['ccc'] = default[modules['type']]

	# print(modules)
# 	m = yaml.load(open('config/modules.yaml', 'r'))
# 	modules.update(m[modules['type']])
	# ex.add_config(yaml.load(open('config/modules.yaml', 'r')))
	# print(ex)

# print(ex.config())
# @ex.config
# def config():
# 	b=1


# def train_with_sacred(config):
# 	# print(_run.config)
# 	modules = yaml.load(open('config/modules.yaml', 'r'))

# 	# ex.add
	# datasets = yaml.load(open('config/datasets.yaml', 'r'))
	# trainer = yaml.load(open('config/trainer.yaml', 'r'))

	# ex.add_config(_run.config['modules'])
	# train(_run.config, ex)

def train(config):
	pass
@ex.main
def train(datasets, modules, trainer):
	pass

	# modules = init_modules(config['modules'])
	# datasets = init_datasets(config['datasets'])
	# trainer = init_trainer(modules, config['trainer'])
	# epochs = tqdm(range(1, config['nepochs'] + 1), ncols=0)
	# for epoch in epochs:
	# 	logs = trainer.train(dataset['train'])
	# 	logs.update(trainer.eval(dataset['train']))
	# 	logs.update(trainer.eval(dataset['test']))

		# if ex:
		# 	for n, log in logs.items():
		# 		if isinstance(log, (int, float)):
		# 			ex.log_scalar(n, log, epoch)
		# 		if torch.is_tensor(log):



	# bar_epoch = tqdm(range(1, nepochs + 1), ncols=0)
	# for epoch in bar_epoch:
	# 	for split, ds in datasets.items():

	# 		torch.set_grad_enabled(split == 'train')
	# 		dl_bar = tqdm(ds, ncols=0, total=min(niter[split], len(ds)))

	# 		trainer.forward(ds, config['niter']['split'])
	# 		trainer.
	# 		iter = 0
	# 		for batch in dl_bar:
	# 			iter += 1
	# 			for k in batch:
	# 				batch[k] = batch[k].to(device)

	# 			logs = trainer.forward(batch)

	# 			if split == 'train':
	# 				logs.update(trainer.backward())

	# 			if iter > niter[split]:
	# 				break


# def populate_args(key, args, default_args)
# 	new_config = default_configs['modules'][config['type']].copy()
# 	new_config.update(config['args'])

# 	config[key]['args'] = new_config

if __name__ == '__main__':
	from sacred.config import load_config_file
	config = load_config_file('config.yaml')

	modules_default_configs = load_config_file('config/modules.yaml')
	for module_name, module_config in config['modules'].items():
		typ, args = module_config['type'], module_config['args']
		new_args = modules_default_configs[typ]
		new_args.update(args)
		module_config['args'] = new_args

	datasets_default_configs = load_config_file('config/datasets.yaml')
	for dataset_name, dataset_config in config['datasets'].items():
		typ, args = dataset_config['type'], dataset_config['args']
		new_args = datasets_default_configs[typ]
		new_args.update(args)
		dataset_config['args'] = new_args

	trainer_default_configs = load_config_file('config/trainer.yaml')
	typ, args = trainer_config['type'], trainer_config['args']
	new_args = trainer_default_configs[typ]
	new_args.update(args)
	trainer_config['args'] = new_args

	# k, args = config['trainer']['type'], v['args']
	# new_config = default_configs['modules'][k]
	# new_config.update(v['args'])
	# config['modules'][n]['args'] = new_config

	import sys
	print(sys.argv)


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


