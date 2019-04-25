from functools import partial

import torch
import torch.nn as nn
from torch.nn import init

from src.utils.misc import pretty_wrap


def init_weights(net, name='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if name == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif name == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif name == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif name == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method {name} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = partial(
                nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
                'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def num_params(module):
    num_params = 0
    for param in module.parameters():
        num_params += param.numel()
    return num_params


def print_net(name, net, init_name, init_gain):
    s = f'Class: {net.__class__.__name__:}\n' \
        f'Init: {init_name}, Gain={init_gain}\n' \
        f'Number of parameters : {num_params(net) / 1e6:.3f}\n'
    print(pretty_wrap(title=name, text=s))


def get_z_random(batchSize: int, nz: int, device: str, random_type: str = 'gauss'):
    if random_type == 'uni':
        z = torch.rand(batchSize, nz, device=device) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batchSize, nz, device=device)
    elif random_type == 'gauss_conjugate':
        std = (torch.randn(batchSize, nz, device=device) * 0.5).exp()
        mean = torch.randn(batchSize, nz, device=device)
        z = torch.cat([mean, std], 1)
    return z


class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'uniform':
            self.uniform_(from_=-1, to=1)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def mod_to_gpu(mod, gpu_id, device):
    if type(gpu_id) == int:
        gpu_id = [gpu_id]
    if len(gpu_id) > 1:
        assert (torch.cuda.is_available())
        mod = torch.nn.DataParallel(mod, gpu_id).cuda()  # multi-GPUs
    else:
        mod = mod.to(device)
    return mod