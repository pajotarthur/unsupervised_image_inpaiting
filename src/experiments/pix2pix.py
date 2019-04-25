import torch
from ignite.metrics import RunningAverage
from torch import nn
from torchvision.utils import make_grid

import src.utils.fid.fid as metrics
from src.utils.net import set_requires_grad
from .GAN_experiment import GAN_experiment


# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(self, x_a, x_b=None, label=None, train_G=False):
        # If training G, enable grad tape

        # print('x_a',x_a[0:1,0:1, :5, :5])
        # print('x_b',x_b[0:1,0:1, :5, :5])

        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            x_b_hat = self.G(x_a, label)

        # print('x_b_hat',x_b_hat[0:1,0:1, :5, :5])


        if train_G:
            fake_ab = torch.cat([x_a, x_b_hat], 1)
            pred_fake = self.D(fake_ab.detach(), label)
            return pred_fake, x_b_hat
        else:
            fake_ab = torch.cat([x_a, x_b_hat], 1)
            real_ab = torch.cat([x_a, x_b], 1)
            pred_real = self.D(real_ab, label)
            pred_fake = self.D(fake_ab, label)

            return pred_real, pred_fake


class Pix2pix(GAN_experiment):
    def __init__(self, lambda_L1=1, **kwargs):
        super(Pix2pix, self).__init__(**kwargs)
        self.GD = G_D(self.gen, self.dis)
        self.lambda_L1 = lambda_L1
        self.criterionL1 = nn.L1Loss()

        if self.lambda_L1 == 0:
            RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_gen_l1'].item()).attach(self.trainer, 'gen_l1')
        else:
            RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_gen_l1'].item() / float(self.lambda_L1)).attach(self.trainer, 'gen_l1')

    def optimize(self, x, y, label, **kwargs):

        self.x_a, self.x_b = y, x

        self.optim_gen.zero_grad()
        self.optim_dis.zero_grad()
        set_requires_grad(self.dis, True)
        set_requires_grad(self.gen, False)

        if self.fp16:
            self.x_a = self.x_a.half()

        for step_index in range(self.num_D_step):
            for accumulation_index in range(self.num_D_acc):

                pred_real, pred_fake = self.GD(x_a=self.x_a, x_b=self.x_b, label=label,
                                               train_G=False)
                loss_dis_real, loss_dis_fake = self.gan_loss.D_loss(pred_fake, pred_real)
                loss_dis = (loss_dis_real + loss_dis_fake) / float(self.num_D_acc)
                loss_dis.backward()
            self.optim_dis.step()

        set_requires_grad(self.dis, False)
        set_requires_grad(self.gen, True)

        self.optim_gen.zero_grad()
        for accumulation_index in range(self.num_G_acc):
            pred_fake, self.x_b_hat = self.GD(x_a=self.x_a, x_b=None, label=label, train_G=True)
            loss_gen_gan = self.gan_loss.G_loss(pred_fake)
            loss_gen_l1 = self.criterionL1(self.x_b_hat, self.x_b.detach()) * self.lambda_L1
            loss_gen = (loss_gen_gan + loss_gen_l1) / float(self.num_G_acc)
            loss_gen.backward()
        self.optim_gen.step()

        return dict(loss_gen=loss_gen,
                    loss_gen_l1=loss_gen_l1,
                    loss_dis_real=loss_dis_real,
                    loss_dis_fake=loss_dis_fake)

    def write_image(self, engine, dataset_name):
        iteration = self.trainer.state.iteration

        b = engine.state.batch

        x = b['x'].cpu()
        y = b['y'].cpu()
        perm = torch.randperm(x.size(0))
        idx = perm[:8]
        x = self.x_a[idx]
        y = self.x_b[idx]

        x_hat = self.x_b_hat[idx]
        img_tensor = torch.cat([x.cpu(), y.cpu(), x_hat.cpu()], dim=0)

        img = make_grid(
                img_tensor,
                nrow=self.num_samples, scale_each=True, normalize=True,
                )

        try:
            self.writers.add_image(dataset_name, img, iteration)
        except:
            self.pbar.log_message('IMPOSSIBLE TO SAVE')

    def compute_fid(self, iteration):
        self.evaluating()
        fake_list, real_list = [], []
        with torch.no_grad():
            for i, batch in enumerate(self.eval):
                true = batch['x'].cuda()
                fake = self.gen(x=batch['y'].cuda())
                if self.fp16:
                    true = true.float()
                    fake = fake.float()

                fake_list.append((fake.cpu() + 1.0) / 2.0)
                real_list.append((true.cpu() + 1.0) / 2.0)

        fake_images = torch.cat(fake_list)
        real_images = torch.cat(real_list)
        mu_fake, sigma_fake = metrics.calculate_activation_statistics(
                fake_images, self.model, self.train.batch_size, device=self.device
                )
        mu_real, sigma_real = metrics.calculate_activation_statistics(
                real_images, self.model, self.train.batch_size, device=self.device
                )
        self.fid_score = metrics.calculate_frechet_distance(
                mu_fake, sigma_fake, mu_real, sigma_real
                )

        if self.writers is not None:
            self.writers.add_scalar('FID', self.fid_score, iteration)
