# from src.experiments.ambientgan import AmbientGan
# from src.experiments.gan import GanExperiment
# from src.experiments.inputation import InputExperiment
# from src.experiments.inputation_paired import InputExperimentPaired
# from src.experiments.inputation_unpaired import InputExperimentUnpaired
# from src.experiments.pggan import PGGanExperiment
# from src.experiments.pluralistic import PluralExperiment
# from src.experiments.unpaired import UnpairedExperiment
from src.experiments.GAN_experiment import GAN_experiment
from src.experiments.ambient_gan import AmbientGan
from src.experiments.pix2pix import Pix2pix
from src.experiments.unsupervised_image_inpaiting import UnsuperviedImageInpainting

def get_experiment_by_name(name):
    # if name == 'unpaired':
    #     return UnpairedExperiment
    # if name == 'gan':
    #     return GanExperiment
    # if name == 'pggan':
    #     return PGGanExperiment
    # if name == 'input':
    #     return InputExperiment
    # if name == 'input_paired':
    #     return InputExperimentPaired
    # if name == 'input_unpaired':
    #     return InputExperimentUnpaired
    # if name == 'pluralistic':
    #     return PluralExperiment
    # if name == 'ambient':
    #     return AmbientGan
    if name == 'gan_expe':
        return GAN_experiment
    if name == 'ambient_gan':
        return AmbientGan
    if name == 'pix2pix':
        return Pix2pix
    if name == 'unsup_image_inpainting':
        return UnsuperviedImageInpainting
    raise NotImplementedError(name)


def init_experiment(_name, **kwargs):
    return get_experiment_by_name(_name)(**kwargs)
