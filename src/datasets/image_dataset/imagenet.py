from pathlib import Path

import torchvision
from torchvision.datasets import ImageFolder


def init_imagenet(root: str, tiny: bool = True, im_size: int = 64, train='train'):
    if train == 'train':
        root = Path(root) / Path('train')
    elif train == 'test':
        root = Path(root) / Path('val')
    elif train == 'train+val':
        root = Path(root) / Path('train+val')
    elif train == 'val':
        root = Path(root) / Path('val')
    else:
        raise FileNotFoundError('not a good train argument')

    normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if tiny:
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize,
                ])
    else:
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(im_size * 1.14)),
                torchvision.transforms.CenterCrop(im_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
                ])
    return ImageFolder(root, transforms)
