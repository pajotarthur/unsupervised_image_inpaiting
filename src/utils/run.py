from os.path import join
from copy import deepcopy
from sacred import Experiment
from sacred.utils import recursive_update
from sacred.config import load_config_file


# functions for generalizing ingredients

# def nested_update(d, sub_d, *keys):
#     if keys and d:
#         element = keys[0]
#         if element:
#             value = d.get(element)
#             if len(keys) == 1:
#                 recursive_update(value, sub_d)
#             else:
#                 return get_nested(value, sub_d, *keys[1:])

# def get_nested(d, *keys):
#     if keys and d:
#         element  = keys[0]
#         if element:
#             value = d.get(element)
#             if len(keys) == 1:
#                 return value
#             else:
#                 return get_nested(value, *keys[1:])

# sort of generic version of the config hook ... maybe for after...
# def default_config(config, command_name, logger):
#     default_config = {}
#     for ingredient in ['modules', 'datasets', 'experiment']:
#         fn = join(default_configs_root, ingredient + '.yaml')
#         default_ingredient_args = load_config_file(fn)
#         ingredient_config = config[ingredient]
#         # if there is only one component per-ingredient,
#         # like in experiment config
#         if 'name' in ingredient_config:
#             ingredient_config = {ingredient: ingredient_config}
#         default_config[ingredient] = {}
#         for component, component_config in ingredient_config.items():
#             default_config[ingredient][component] = {}
#             name = component_config['name']
#             if 'name' in config[ingredient]:
#                 default_config[ingredient]['args'] = default_ingredient_args[name]
#             else:
#                 default_config[ingredient][component]['args'] = default_ingredient_args[name]
#     return default_config


def sacred_run(command, name='train', default_configs_root='default_configs'):

    ex = Experiment(name)



    def default_config(config, command_name, logger):
        config = config['config']
        default_config = {}
        
        # loading modules default config
        fn = join(default_configs_root, 'modules.yaml')
        default_modules_args = load_config_file(fn)
        default_config['modules'] = {}
        for module, module_config in config['modules'].items():
            default_config['modules'][module] = {}
            default_args = default_modules_args[module_config['name']]
            default_config['modules'][module]['args'] = default_args

        # loading datasets default config
        fn = join(default_configs_root, 'datasets.yaml')
        default_datasets_args = load_config_file(fn)
        default_config['datasets'] = {}
        for dataset, dataset_config in config['datasets'].items():
            default_config['datasets'][dataset] = {}
            default_args = default_datasets_args[dataset_config['name']]
            default_config['datasets'][dataset]['args'] = default_args

        # loading experiment default configs
        fn = join(default_configs_root, 'experiment.yaml')
        default_experiment_args = load_config_file(fn)
        default_config['experiment'] = {}
        default_args = default_experiment_args[config['experiment']['name']]
        default_config['experiment']['args'] = default_args

        return {'config': default_config}

    ex.config_hook(default_config)
    ex.main(command)
    ex.run_commandline()