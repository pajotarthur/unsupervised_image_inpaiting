import torch
import torch.optim as optim
import torch.nn.functional as F

from src.utils.metrics import Metrics
from .base import EpochExperiment


class MNISTExperiment(EpochExperiment):

    def __init__(self, model, optim=None, lr_scheduler=None, trainset=[], valset=[], testset=[], **kwargs):
        super(MNISTExperiment, self).__init__(**kwargs)
        self.model = model
        self.train = trainset
        self.valset = valset
        self.testset = testset
        self.optim = optim

    def update_state(self, epoch):
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_lr()
        assert(len(lr) == 1)
        lr = lr[0]
        return {'lr': lr}

    def init_metrics(self, *args, **kwargs):
        m = Metrics(*args, **kwargs)
        for name, _ in self.named_datasets():
            m.Parent(name=name,
                children=(m.AvgMetric(name='loss'),
                          m.AvgMetric(name='acc'))
            )
        m.Parent(name='state',
            children=(m.AvgMetric(name='lr'),),
        )
        return m

    def __call__(self, input, target, train=True, evaluate=True):
        self.train_mode(train)
        output = self.model(input)
        loss = F.nll_loss(output, target)

        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        results = {}
        if evaluate:
            with torch.no_grad():
                self.train_mode(False)
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.view_as(pred)).sum()
                results['acc'] = correct

        return {
            'loss': loss,
            **results,
        }