
from os.path import join
from sacred.config import load_config_file


def fill_config_with_default(config, default_root):
	for component in ['modules', 'datasets', 'trainer']:
		default_configs_fn = join(default_root, component + '.yaml')
		default_configs = load_config_file(default_configs_fn)
		name = config[component]['name'], 
		args = config[component]['args']
		new_args = default_configs.copy()
		new_args.update(args)
		config[component]['args'] = new_args

def init_config(default_root='default_configs'):
	config = parse_commandline()
	fill_config_with_default(config)
	return config
