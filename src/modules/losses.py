import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()

        self.gan_mode = gan_mode
        # self.target_real_label = target_real_label
        # self.target_fake_label = target_fake_label

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()

    def D_loss(self, dis_fake, dis_real):
        if self.gan_mode == 'hinge':
            loss_real = torch.mean(self.loss(1. - dis_real))
            loss_fake = torch.mean(self.loss(1. + dis_fake))
            return loss_real, loss_fake
        elif self.gan_mode in ['vanilla', 'lsgan']:
            loss_real = self.loss(dis_real, self.real_label.expand_as(dis_real).type_as(dis_real))
            loss_fake = self.loss(dis_fake, self.fake_label.expand_as(dis_fake).type_as(dis_fake))
            return loss_real, loss_fake
        raise NotImplementedError

    def G_loss(self, dis_fake):
        if self.gan_mode == 'hinge':
            loss = -torch.mean(dis_fake)
        elif self.gan_mode in ['vanilla', 'lsgan']:
            loss = self.loss(dis_fake, self.real_label.expand_as(dis_fake).type_as(dis_fake))

        return loss
