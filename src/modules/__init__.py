from .utils import init_weights, print_net
from .gen_factory import Generator
from .dis_factory import Discriminator
from .pix2pix_factory import ResnetGenerator

def get_module_by_name(name):
    if name == 'biggan_gen':
        return Generator
    if name == 'biggan_dis':
        return Discriminator
    if name == 'pix2pix':
        return ResnetGenerator
    if name is None:
        return None
    raise NotImplementedError(name)


def init_module(_name, init_name=None, init_gain=None, gpu_id=[], **kwargs):
    """Only works for network modules"""
    # print(kwargs)
    module = get_module_by_name(_name)(**kwargs)
    if (init_name is not None) and (init_gain is not None):
        init_weights(module, init_name, init_gain)
    print_net(_name, module, init_name, init_gain)

    return module
