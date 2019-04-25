import pprint

from torch.utils.data.sampler import Sampler

from src.utils.misc import pretty_wrap


class SubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def dataset_length(dl):
    bs = dl.batch_size
    ns = len(dl.batch_sampler.sampler)
    length = (len(dl) - 1) * bs + (ns % bs == 0) * bs
    if not dl.drop_last:
        length += ns % bs
    return length


def print_dataset(name, dl, config):
    r"""
    Arguments:
        dl (torch.utils.data.DataLoader): dataset to print
    """
    name = '{:^80s}'.format(name)
    s = 'Class: {}\n'.format(dl.dataset.__class__.__name__)
    s += 'Length: Effective {}, Dataset: {}\n' \
        .format(dataset_length(dl), len(dl.dataset))
    s += 'Batch size: {}\n'.format(dl.batch_size)
    s += 'Config: \n'
    s += pprint.pformat(config)
    print(pretty_wrap(title=name, text=s, width=180))
