
import torch
from tqdm import tqdm
from sacred import Experiment

from src.trainers import init_trainer
from src.datasets import init_dataset
from src.modules import init_module
# from src.utils import init_config

from sacred.utils import ensure_wellformed_argv



def train(_run, modules, trainers, datasets, exp_name, nepochs, ngpu):
    config = {
        'modules': modules,
        'trainer': trainer,
        'datasets': datasets,
    }

    print('test!')
    return
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

    # config = init_config()

    from src.utils import ExperimentNew

    ex = ExperimentNew('test')
    ex.main(train)
    ex.run_commandline()

    # exp = Experiment('parser')
    # from docopt import docopt
    # import sys
    # from sacred.arg_parser import format_usage, get_config_updates

    # argv = sys.argv
    # # print('U', exp.get_usage())
    # argv = ensure_wellformed_argv(argv)
    # short_usage, usage, internal_usage = exp.get_usage()
    # args = docopt(internal_usage, [str(a) for a in argv[1:]], help=False)
    # cmd_name = args.get('COMMAND') or default_command
    
    # config_updates, named_configs = get_config_updates(args['UPDATE'])

    # err = exp._check_command(cmd_name)
    # if not args['help'] and err:
    #     print(short_usage)
    #     print(err)
    #     exit(1)

    # if exp._handle_help(args, usage):
    #     exit()
    # from src.utils import fill_config_with_default

    # config = {}
    # for named_config in named_configs:
    #     if named_config.endswith('.yaml'):
            # print(named_config)




    # print('config', config_updates, named_configs)
    # exp.main(train)
    # exp.run(cmd_name, config_updates, named_configs, {}, args)
 
    # config_updates, named_configs = get_config_updates(args['UPDATE'])
    # print(args)
    # ex = Experiment('test')
    # ex.main(train)
    # config = {}
    # ex.add_config({'config': config})
    # ex.run_commandline()
    # ex.run('train')
    # ex = Experiment(config['exp_name'])
    # ex.main(main)
    # ex.run('main')#config_updates={'config':config})
    # import ipdb; ipdb.set_trace() # BREAKPOINT


