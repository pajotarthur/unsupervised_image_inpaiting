from collections import OrderedDict
import torch

class Meter():
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, value, n=1):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def __repr__(self):
        return '{:.5f} ({:.5f})'.format(self.value, self.avg)


class ScalarMeter(Meter):
    def reset(self):
        self.avg = -1

    def update(self, value):
        if torch.is_tensor():
            value = value.item()
        self.avg = value

    def get(self):
        return self.avg


class AverageMeter(Meter):
    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n= 1):
        if torch.is_tensor():
            value = value.item()
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self):
        return self.avg

# class BestMeter(Meter):

class Meters(OrderedDict):
    def __init__(self, name, children):
        self.name = name
        self.update(children)

    def update(self, n=1, **kwargs):
        for name, value in kwargs.items():
            self[name].update(value)

    def __str__(self):
        s = ''
        for name, meter in self.items():
            print('ok')
            s += name
        return s

if __name__ == "__main__":

    meters = Meters('train', children={'loss': AverageMeter(), 'prec': AverageMeter()})
    print(meters)

