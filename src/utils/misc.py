import os
from datetime import datetime
from time import sleep

import numpy as np
import prettytable as pt
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage


def fix_seed(seed):
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def pretty_wrap(text, title=None, width=80):
    table = pt.PrettyTable(
            header=title is not None,
            )
    table.field_names = [title]
    for t in text.split('\n'):
        for i in range(0, len(t), width):
            table.add_row([t[i: i + width]])

    return table


def make_basedir(root, timestamp=False, attempts=5):
    """Takes 5 shots at creating a folder from root,
    adding timestamp if desired.
    """
    for i in range(attempts):
        basedir = root
        if timestamp is None:
            timestamp = datetime.now().strftime("%a-%b-%d-%H:%M:%S.%f")
            basedir = os.path.join(basedir, timestamp)
        try:
            os.makedirs(basedir)
            return basedir
        except:
            sleep(0.01)
    raise FileExistsError(root)


def init_checkpoint_handler(dirname, filename_prefix, metric_name=None, save_interval=1, higher_is_better=True,
                            score_function=None, score_name=None, **kwargs):
    if metric_name is not None:
        assert (score_function is None)
        assert (score_name is None)

        def metric_to_score_func(engine, metric_name):
            metrics = engine.state.metrics
            return metrics.get(metric_name, -np.inf)

        score_function = lambda engine: metric_to_score_func(engine, metric_name)
        score_name = metric_name

    if score_function is not None:
        final_score_function = lambda engine: -score_function(engine)
    else:
        final_score_function = score_function

    return ModelCheckpoint(
            dirname, filename_prefix,
            score_function=final_score_function,
            score_name=score_name,
            save_interval=save_interval,
            **kwargs
            )


class RunningAverageSkipNone(RunningAverage):
    """does not compute value when value is None"""

    def compute(self):
        if self._value is None:
            self._value = self._get_src_value()
        else:
            if self._get_src_value() is not None:
                self._value = self._value * self.alpha + (1.0 - self.alpha) * self._get_src_value()
        return self._value
