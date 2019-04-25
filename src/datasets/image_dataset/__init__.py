from functools import partial

from .celebA import CelebA
from .ffhq import FFHQ
from .imagenet import init_imagenet
from .lsun import LSUN
from .recipe import Recipe


def get_dataset_by_name(name):
    if name == 'imagenet_tiny':
        return partial(init_imagenet, tiny=True)
    elif name == 'imagenet':
        return partial(init_imagenet, tiny=False)
    elif name == 'celebA':
        return CelebA
    elif name == 'ffhq':
        return FFHQ
    elif name == 'recipe':
        return Recipe
    elif name == 'lsun':
        return LSUN

    raise NotImplementedError(name)
