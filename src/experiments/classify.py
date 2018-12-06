import torch.optim as optim
import torch.nn.functional as F

from src.utils.metrics import Metrics
from .base import EpochExperiment


class MNISTExperiment(EpochExperiment):

    def __init__(self, model, optim, train=[], val=[], test=[], **kwargs):
        super(MNISTExperiment, self).__init__(**kwargs)
        self.model = model
        self.train = train
        self.val = val
        self.test = test
        self.optim = optim

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

    def __call__(self, input, target, mode='train+eval'):
        self.train_mode('train' in mode)
        output = self.model(input)
        loss = F.nll_loss(output, target)

        if 'train' in mode:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        eval_results = {}
        if 'eval' in mode:
            with torch.no_grad():
                self.train_mode(False)
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.view_as(pred)).sum()
                eval_results['acc'] = correct

        return {
            'loss': loss,
            **eval_results,
        }