import torch.optim as optim


def get_optim_by_name(name):
    if name == 'sgd':
        return optim.SGD
    elif name == 'adam':
        return optim.Adam
    raise NotImplementedError(name)

def init_optimizer(modules, _name, _modules, **kwargs):
    if isinstance(_modules, str):
        _modules = [_modules]
    parameters = []
    for name in _modules:
        module = modules[name]
        parameters += list(module.parameters())
    optim = get_optim_by_name(_name)(
            parameters,
            **kwargs
        )
    return optim