import time
from collections import defaultdict, OrderedDict

from ._metrics import TimeMetric_, AvgMetric_, SumMetric_, Parent_,\
    SimpleMetric_, BestMetric_, DynamicMetric_


class Metrics(object):

    def __init__(self, name=None, time_indexing=True, xlabel=None):
        """ Create an experiment with the following parameters:
        - time_indexing (bool): use time to index values (otherwise counter)
        """
        self.name = name
        self.date_and_time = time.strftime('%d-%m-%Y--%H-%M-%S')
        self.metrics = defaultdict(dict)
        self.time_indexing = time_indexing

    def NewMetric_(self, name, Metric_, time_idx, **kwargs):
        if time_idx is None:
            time_idx = self.time_indexing
        metric = Metric_(name,
                         time_idx=time_idx, **kwargs)
        self.register_metric(metric)
        return metric

    def AvgMetric(self, name, time_idx=None):
        return self.NewMetric_(name,  AvgMetric_, time_idx)

    def SimpleMetric(self, name, time_idx=None):
        return self.NewMetric_(name, SimpleMetric_, time_idx)

    def TimeMetric(self, name):
        return self.NewMetric_(name, TimeMetric_, False)

    def SumMetric(self, name, time_idx=None):
        return self.NewMetric_(name, SumMetric_, time_idx)

    def BestMetric(self, name, mode="max", time_idx=None):
        return self.NewMetric_(name, BestMetric_, time_idx, mode=mode)

    def DynamicMetric(self, name, fun=None, time_idx=None):
        return self.NewMetric_(name, DynamicMetric_, time_idx, fun=fun)

    def Parent(self, name, children=()):
        for child in children:
            self.remove_metric(child)
        wrapper = Parent_(name, children=children)
        self.register_metric(wrapper)
        return wrapper

    def register_metric(self, metric):
        self.metrics[metric.name] = metric

    def remove_metric(self, metric):
        self.metrics.pop(metric.name)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def __getattr__(self, name):
        if name in self.metrics:
            return self.metrics[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
