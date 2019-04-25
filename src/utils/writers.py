import os

from PIL import Image


def init_writer(_name, dirname=None, sacred_run=None, **config):
    if _name == 'sacred':
        return SacredWriter(dirname=dirname, sacred_run=sacred_run, **config)
    elif _name == 'tensorboard':
        from tensorboardX import SummaryWriter
        return SummaryWriter(log_dir=dirname, **config)
    elif _name == 'visdom':
        from tensorboardX.visdom_writer import VisdomWriter
        return VisdomWriter()
    raise NotImplementedError(_name)


def init_writers(*writer_kwargs, sacred_run=None, dirname=None):
    writers = []
    for kwargs in writer_kwargs:
        writer = init_writer(sacred_run=sacred_run, dirname=dirname, **kwargs)
        writers.append(writer)
    return WrappingWriters(writers)


class BaseWriter:
    def add_scalar(self, tag, scalar, step):
        raise NotImplementedError

    def add_image(self, tag, image, step, basedir=None):
        raise NotImplementedError

    def add_audio(self, tag, audio, step):
        raise NotImplementedError

    def add_text(self, tag, text, step):
        raise NotImplementedError


class WrappingWriters(BaseWriter):
    def __init__(self, writers):
        self.writers = writers

    def add_scalar(self, tag, scalar, step):
        for w in self.writers:
            w.add_scalar(tag, scalar, step)

    def add_image(self, tag, image, step, basedir=None):
        for w in self.writers:
            w.add_image(tag, image, step)

    def add_audio(self, tag, audio, step):
        for w in self.writers:
            w.add_audio(tag, audio, step)

    def add_text(self, tag, text, step):
        for w in self.writers:
            w.add_text(tag, text, step)

    def finish(self):
        for w in self.writers:
            w.close()


class SacredWriter(BaseWriter):
    def __init__(self, sacred_run, dirname, save_info=False):
        self.sacred_run = sacred_run
        self.dirname = str(dirname)
        self.save_info = save_info

    def add_scalar(self, tag, scalar, step):
        self.sacred_run.log_scalar(tag, scalar, step)

    def add_image(self, tag, image, step):
        ndarr = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        filename = os.path.join(self.dirname, f'{tag}_{step}.png')
        im.save(filename)
        # self.sacred_run.add_artifact(filename, tag)
        #
        # if self.save_info:
        #     if 'image' not in self.sacred_run.info:
        #         self.sacred_run.info['image'] = {}
        #     if step not in self.sacred_run.info['image']:
        #         self.sacred_run.info['image'][step] = {}
        #     self.sacred_run.info['image'][step][tag] = filename

    def close(self):
        return
