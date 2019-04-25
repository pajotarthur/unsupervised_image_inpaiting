import os
from collections import Mapping

from ignite._utils import convert_tensor
from ignite.contrib.handlers import ProgressBar
# from src.utils.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from src.experiments.base import BaseExperiment
from src.utils.misc import make_basedir
from src.utils.writers import init_writers


class DatasetExperiment(BaseExperiment):
    def __init__(self, train, niter, nepoch, eval=None, gpu_id=[0],
                 sacred_run=None, writers=None, root=None, corruption=None,
                 **kwargs):
        super(DatasetExperiment, self).__init__(**kwargs)

        self.train = train
        self.eval = eval

        self.corruption = corruption

        self.niter = niter
        self.nepoch = nepoch
        self.device = 'cuda:0'

        self.sacred_run = sacred_run

        self.gpu_id = gpu_id

        if root is not None:
            self.basedir = make_basedir(os.path.join(root, str(sacred_run._id)))
        else:
            writers = None
            checkpoint = None

        if writers is not None:
            self.writers = init_writers(*writers, sacred_run=sacred_run, dirname=self.basedir)
        else:
            self.writers = None

        self.create_trainer()

    def create_trainer(self):
        self.trainer = Engine(self.train_step)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.K_step)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_metrics, 'train')
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.write_metrics, 'train')

        self.pbar = ProgressBar()

        self.timer = Timer(average=True)
        self.timer.attach(self.trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                          pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.print_times)

    def print_times(self, engine):
        iteration = engine.state.iteration
        if self.writers is not None:
            self.writers.add_scalar('time_per_epoch', self.timer.total, iteration)
            self.writers.add_scalar('time_per_batch', self.timer.value(), iteration)
        self.timer.reset()

    def train_step(self, engine, batch):
        self.training()
        if isinstance(batch, Mapping):
            return self.optimize(**convert_tensor(batch, self.device))
        else:
            raise NotImplementedError

    def infer(self, **kwargs):
        raise NotImplementedError

    def optimize(self, **kwargs):
        raise NotImplementedError

    def K_step(self, **kwargs):
        raise NotImplementedError

    def log_metrics(self, engine, dataset_name):
        iteration = self.trainer.state.iteration
        epoch = self.trainer.state.epoch
        log = f"EPOCH {epoch}" f" - ITER {iteration} - {dataset_name}"

        if self.sacred_run is not None:
            log = f"ID {self.sacred_run._id} - " + log

        for metric, value in engine.state.metrics.items():
            if value is None:
                value = float('nan')
            log += f" {metric}: {value:.6f}"

        log += self.extra_log()
        self.pbar.log_message(log)

    def extra_log(self):
        return ""

    def write_metrics(self, engine, dataset_name):
        if self.writers is not None:
            for metric, value in engine.state.metrics.items():
                name = dataset_name + '/' + metric
                self.writers.add_scalar(name, value, self.trainer.state.iteration)

    def run(self):
        self.trainer.run(self.train, max_epochs=self.nepoch)
