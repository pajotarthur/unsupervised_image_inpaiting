from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
from torch._six import container_abcs, int_classes
from torch.utils.data.dataloader import default_collate


from .kth import KTH


def get_dataset_by_name(name):
    if name == 'kth':
        return KTH

    raise NotImplementedError(name)


def get_output_transform(name):

    return lambda x: x


def sequence_collate(batch):
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        max_seq_len = max(seq.shape[1] for seq in batch)
        batch = [F.pad(seq, (0,0, 0,0, 0,max_seq_len - seq.size(1))) for seq in batch]
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray':
            dim = len(batch[0].shape)
            if dim >= 1:
                return torch.from_numpy(np.concatenate(batch, axis=0))  # we suppose that the first dimension is batch dimension
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: sequence_collate([d[key] for d in batch]) for key in batch[0]}
    raise TypeError('\'{}\' is not supported yet.'.format(elem_type))


def init_dataset(corruption, _name, batch_size, num_workers, drop_last, shuffle, size='all', pin_memory=True, seq=True, **kwargs):
    ds = get_dataset_by_name(_name)(**kwargs)
    t = get_output_transform(_name)

    if corruption is not None:
        ds = CorruptedSequenceDataset(ds, corruption, t)

    if size == 'all':
        size = len(ds)
    else:
        size = size

    if shuffle:
        sampler = SubsetRandomSampler(range(size))
    else:
        sampler = SubsetSequentialSampler(range(size))

    collate_fn = sequence_collate
    if _name == 'moving_mnist' or seq is False:
        collate_fn = default_collate

    dl = DataLoader(dataset=ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                    )
    print_dataset(_name, dl, kwargs)
    return dl