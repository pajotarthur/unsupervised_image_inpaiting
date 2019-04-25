import os

from sacred import Experiment
from sacred import SETTINGS
from sacred.config import load_config_file
from sacred.utils import SacredInterrupt

SETTINGS.HOST_INFO.INCLUDE_GPU_INFO = True
SETTINGS.HOST_INFO.CAPTURED_ENV = [
        'OMP_NUM_THREADS',
        'CUDA_VISIBLE_DEVICES',
        ]


class CustomInterrupt(SacredInterrupt):
    def __init__(self, STATUS):
        print(STATUS)
        self.STATUS = STATUS


def get_component_configs(config, component_name, default_configs_file):
    """
    :param config: The global config given by the user.
    :param component_name: The key of the root-level element to process in the config.
    :param default_configs_file: The path of the file containing the default configs for the current component.
    :return: A dict containing the default configurations for each element under the given component.
    """
    component_configs = {}
    default_configs = load_config_file(default_configs_file)
    if '_name' in config[component_name]:
        specified_config = config[component_name]
        selected_config = specified_config['_name']
        component_configs = default_configs.get(selected_config, {})
    else:
        for name, specified_config in config[component_name].items():
            selected_config = specified_config['_name']
            component_configs[name] = default_configs.get(selected_config, {})
    return component_configs


def sacred_run(command, default_configs_root='configs/default'):
    ex = Experiment('default')

    @ex.config_hook
    def default_config(config, command_name, logger):
        default_config = {}
        for comp, conf in config.items():
            default_file_path = os.path.join(default_configs_root, f'{comp}.yaml')
            default_config[comp] = get_component_configs(config, comp, default_file_path)
        return default_config

    ex.main(command)
    ex.run_commandline()
