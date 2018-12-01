# import sys
# from os.path import join
# from docopt import docopt, printable_usage

# from sacred.config import load_config_file

# from sacred.utils import ensure_wellformed_argv
# from sacred.arg_parser import format_usage, get_config_updates


# def fill_config_with_default(config, default_root):
#   for component in ['modules', 'datasets', 'trainer']:
#       default_configs_fn = join(default_root, component + '.yaml')
#       default_configs = load_config_file(default_configs_fn)
#       name = config[component]['name'], 
#       args = config[component]['args']
#       new_args = default_configs.copy()
#       new_args.update(args)
#       config[component]['args'] = new_args

# def init_config(argv=None, default_root='default_configs'):

#   if argv is None:
#       argv = sys.argv

#   doc = 'usage: lk'
#   from docopt import docopt
#   docopt(doc, [str(a) for a in argv[1:]], help=False)
#   print('ok')
#   cmd_name = args.get('COMMAND')
#   config_updates, named_configs = get_config_updates(args['UPDATE'])
#   print('updates', config_updates, 'named', named_configs)

#   config = {}
#   # config = parse_commandline()
#   # fill_config_with_default(config)
#   return config


from sacred.commandline_options import gather_command_line_options, ForceOption, LoglevelOption


from sacred import Experiment

class ExperimentNew(Experiment):
    def _create_run(self, command_name=None, config_updates=None,
                    named_configs=(), meta_info=None, options=None):
        command_name = command_name or self.default_command
        if command_name is None:
            raise RuntimeError('No command found to be run. Specify a command '
                               'or define a main function.')

        default_options = self.get_default_options()
        if options:
            default_options.update(options)
        options = default_options

        # call option hooks
        for oh in self.option_hooks:
            oh(options=options)

        run = create_run(self, command_name, config_updates,
                         named_configs=named_configs,
                         force=options.get(ForceOption.get_flag(), False),
                         log_level=options.get(LoglevelOption.get_flag(),
                                               None))
        run.meta_info['command'] = command_name
        run.meta_info['options'] = options

        if meta_info:
            run.meta_info.update(meta_info)

        for option in gather_command_line_options():
            option_value = options.get(option.get_flag(), False)
            if option_value:
                option.apply(option_value, run)

        self.current_run = run
        return run

from copy import copy, deepcopy
        

from sacred.utils import (convert_to_nested_dict,
    set_by_dotted_path, iterate_flattened, join_paths)
from sacred.host_info import get_host_info
from sacred.run import Run

from sacred.initialize import (distribute_config_updates,
    gather_ingredients_topological, create_scaffolding,
    initialize_logging, get_scaffolding_and_config_name,
    distribute_presets, get_configuration, recursive_update,
    get_command)

from sacred.initialize import get_config_modifications


def create_run(experiment, command_name, config_updates=None,
               named_configs=(), force=False, log_level=None):

    sorted_ingredients = gather_ingredients_topological(experiment)
    scaffolding = create_scaffolding(experiment, sorted_ingredients)
    # get all split non-empty prefixes sorted from deepest to shallowest
    prefixes = sorted([s.split('.') for s in scaffolding if s != ''],
                      reverse=True, key=lambda p: len(p))

    # --------- configuration process -------------------

    # Phase 1: Config updates
    config_updates = config_updates or {}
    config_updates = convert_to_nested_dict(config_updates)
    root_logger, run_logger = initialize_logging(experiment, scaffolding,
                                                 log_level)
    distribute_config_updates(prefixes, scaffolding, config_updates)

    # Phase 2: Named Configs
    for ncfg in named_configs:
        scaff, cfg_name = get_scaffolding_and_config_name(ncfg, scaffolding)
        scaff.gather_fallbacks()
        ncfg_updates = scaff.run_named_config(cfg_name)
        distribute_presets(prefixes, scaffolding, ncfg_updates)
        for ncfg_key, value in iterate_flattened(ncfg_updates):
            set_by_dotted_path(config_updates,
                               join_paths(scaff.path, ncfg_key),
                               value)

    distribute_config_updates(prefixes, scaffolding, config_updates)

    # # Phase 3: Normal config scopes
    # for scaffold in scaffolding.values():
    #     scaffold.gather_fallbacks()
    #     scaffold.set_up_config()

    #     # update global config
    #     config = get_configuration(scaffolding)
    #     # run config hooks
    #     config_hook_updates = scaffold.run_config_hooks(
    #         config, command_name, run_logger)
    #     recursive_update(scaffold.config, config_hook_updates)

    # Phase 4: finalize seeding
    for scaffold in reversed(list(scaffolding.values())):
        scaffold.set_up_seed()  # partially recursive

    config = get_configuration(scaffolding)
    config_modifications = get_config_modifications(scaffolding)

    # ----------------------------------------------------

    experiment_info = experiment.get_experiment_info()
    host_info = get_host_info()
    main_function = get_command(scaffolding, command_name)
    pre_runs = [pr for ing in sorted_ingredients for pr in ing.pre_run_hooks]
    post_runs = [pr for ing in sorted_ingredients for pr in ing.post_run_hooks]

    run = Run(config, config_modifications, main_function,
              copy(experiment.observers), root_logger, run_logger,
              experiment_info, host_info, pre_runs, post_runs,
              experiment.captured_out_filter)

    if hasattr(main_function, 'unobserved'):
        run.unobserved = main_function.unobserved

    run.force = force

    for scaffold in scaffolding.values():
        scaffold.finalize_initialization(run=run)

    return run

