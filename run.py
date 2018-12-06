from src.experiments import init_experiment
from src.datasets import init_dataset
from src.modules import init_module
from src.utils.run import sacred_run


def init_and_run(experiment, modules, datasets, _run=None):

    # initializing datasets
    dsets = {}
    for dataset_name, dataset_config in datasets.items():
        dsets[dataset_name] = init_dataset(**dataset_config)

    # initializing modules
    mods = {}
    for module_name, module_config in modules.items():
        mods[module_name] = init_module(**module_config)

    # initializing experiment and running it
    init_experiment(**mods, **dsets, **experiment).run()


if __name__ == '__main__':
    sacred_run(init_and_run, name='testing!')