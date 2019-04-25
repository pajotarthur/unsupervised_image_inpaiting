from .corruptions_static import *


def init_corruption(_name, **kwargs):
    print(_name)
    if _name == 'none':
        return None
    elif _name == 'keep_patch':
        return KeepPatch(**kwargs)
    elif _name == 'remove_pix':
        return RemovePixel(**kwargs)
    elif _name == 'remove_pix_dark':
        return RemovePixelDark(**kwargs)
    elif _name == 'conv_noise':
        return ConvNoise(**kwargs)
    elif _name == 'vertical_bar':
        return MovingVerticalBar(**kwargs)
    raise NotImplementedError(_name)