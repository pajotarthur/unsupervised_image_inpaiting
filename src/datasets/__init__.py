from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial

from .mnist import MNIST, FashionMNIST


def get_dataset_by_name(name):
    if name == 'mnist':
        return partial(mnist, False)
    elif name == 'fashion_mnist':
        return partial(mnist, True)
    raise NotImplementedError(name)


def mnist(fashion=False, *args, **kwargs):
    transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
    ])
    if fashion:
        cl = FashionMNIST
    else:
        cl = MNIST
    return cl(transform=transform, *args, **kwargs)


def init_dataset(name, args):
    ds_args = args.copy()
    batch_size = ds_args.pop('batch_size')
    num_workers = ds_args.pop('num_workers')
    drop_last = ds_args.pop('drop_last')
    shuffle = ds_args.pop('shuffle')
    ds = get_dataset_by_name(name)(**ds_args)
    dl = DataLoader(dataset=ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    )
    return 