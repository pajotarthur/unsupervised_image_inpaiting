from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.datasets.experiment.corrupted import CorruptedDataset
from ._utils import SubsetSequentialSampler, print_dataset
from .experiment.corrupted import CorruptedDataset
from .experiment.progressive import ProgressiveDataset
from .image_dataset import get_dataset_by_name


def init_dataset(_name, batch_size, num_workers, drop_last, shuffle, size='all', pin_memory=True, corruption=None, progressive=False,
                 **kwargs):
    ds = get_dataset_by_name(_name)(**kwargs)

    if corruption is not None:
        ds = CorruptedDataset(ds, corruption=corruption)

    if progressive:
        ds = ProgressiveDataset(ds)
    if size == 'all':
        size = len(ds)

    if shuffle:
        sampler = SubsetRandomSampler(range(size))
    else:
        sampler = SubsetSequentialSampler(range(size))

    dl = DataLoader(dataset=ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    sampler=sampler,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                    )

    print_dataset(_name, dl, kwargs)

    return dl
