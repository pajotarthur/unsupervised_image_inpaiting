import torch.optim as optim

from .base2 import EpochExperiment2


class MNISTExperiment2(EpochExperiment2):

    def __init__(*args, momentum=0.9):

        EpochExperiment2.__init__(self, *args)
        self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def init_metrics(self):
        if self.train != []:
            self.add_metrics(name='train',
                children=(meters.AverageMeter(), )
            )

    def __call__(self, batch, mode='train+eval'):
        self.train('train' in mode)

        for b in batch.values():
                b.to(self.device)
        output = self.model(batch['sample'])
        loss = F.nll_loss(output, batch['class'])

        if 'train' in mode:
                self.optim.zero_grad()
                self.loss.backward()
                self.optim.step()

        eval_results = {}
        if 'eval' in mode:
                self.eval()
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.view_as(pred)).sum()
                eval_results['correct'] = correct

        return {
                'loss': loss,
                **eval_results,
        }
