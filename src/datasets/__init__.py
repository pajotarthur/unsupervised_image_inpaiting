from torch.utils.data import DataLoader


from .mnist import MNIST, FashionMNIST


def get_dataset_by_name(name):
    if name == 'mnist':
        return MNIST
    elif name == 'fashion_mnist':
        return FashionMNIST
    else:
        raise NotImplementedError(name)


def init_dataset(config):
    name, args = config['name'], config['args']
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
    return dl