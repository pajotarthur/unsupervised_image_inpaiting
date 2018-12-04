from src.experiments import init_experiment
from src.datasets import init_dataset
from src.modules import init_module
from src.run import sacred_run


def init_and_run(config, _run=None):

    # initializing datasets
    datasets = {}
    for dataset_name, dataset_config in config['datasets'].items():
        datasets[dataset_name] = init_dataset(**dataset_config)

    # initializing modules
    modules = {}
    for module_name, module_config in config['modules'].items():
        modules[module_name] = init_module(**module_config)

    # initializing experiment and running it
    init_experiment(**modules, **datasets, **config['experiment']).run()


if __name__ == '__main__':
    sacred_run(init_and_run, name='testing!')