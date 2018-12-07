from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial

from .mnist import mnist


def get_dataset_by_name(name):
    if name == 'mnist':
        return partial(mnist, fashion=False)
    elif name == 'fashion_mnist':
        return partial(mnist, fashion=True)
    raise NotImplementedError(name)


def init_dataset(_name, batch_size, num_workers, drop_last, shuffle, **kwargs):
    ds = get_dataset_by_name(_name)(**kwargs)
    dl = DataLoader(dataset=ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    )
    return dl