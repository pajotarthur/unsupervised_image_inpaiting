import torch
from ignite.metrics import RunningAverage
from torch import nn
from torchvision.utils import make_grid

import src.utils.fid.fid as metrics
from src.utils.net import set_requires_grad, mod_to_gpu
from .GAN_experiment import GAN_experiment


# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D_E(nn.Module):
    def __init__(self, G, D, E, corruption, lambda_z=0, lambda_L1=0, lambda_gen_2=0):
        super(G_D_E, self).__init__()
        self.G = G
        self.D = D
        self.E = E
        self.corruption = corruption
        self.lambda_z = lambda_z
        self.lambda_L1 = lambda_L1
        self.lambda_gen_2 = lambda_gen_2

    def forward(self, y, x=None, z=None, theta=None, label=None, train_G=False):
        # If training G, enable grad tape

        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            x_hat_orig = self.G(y, z, label)
            x_hat_mask, _ = self.corruption(x_hat_orig, theta=1 - theta)
            x_hat = x_hat_mask + y
            y_tilde, theta2 = self.corruption(x_hat)

        if train_G:
            if self.lambda_z > 0:
                z_hat = self.E(x_hat)
            else:
                z_hat = None
            if self.lambda_L1 > 0.0 or self.lambda_gen_2 > 0:
                z_recon = self.E(y)
                x_recon = self.G(y_tilde, z_recon, label=label)
                y_recon, _ = self.corruption(x_recon, theta=theta)
                if self.lambda_gen_2:
                    pred_fake_recon = self.D(y_recon, label)
                else:
                    pred_fake_recon = None
            else:
                pred_fake_recon = None
                y_recon = None

            pred_fake = self.D(y_tilde, label)

            return pred_fake, pred_fake_recon, y_recon, z_hat, x_hat, y_tilde
        else:
            pred_real = self.D(y, label)
            pred_fake = self.D(y_tilde.detach(), label)

            return pred_real, pred_fake


class UnsuperviedImageInpainting(GAN_experiment):
    def __init__(self, lambda_z=0, lambda_kl=0, lambda_L1=0, lambda_gen_2=0, use_l1=True, enc=None, **kwargs):
        super(UnsuperviedImageInpainting, self).__init__(**kwargs)

        self.lambda_z = lambda_z
        self.lambda_kl = lambda_kl
        self.lambda_L1 = lambda_L1
        self.lmbda_gen_2 = lambda_gen_2

        # self.enc = mod_to_gpu(enc, self.gpu_id, self.device)

        self.GDE = G_D_E(self.gen, self.dis, enc, corruption=self.corruption, lambda_z=self.lambda_z, lambda_L1=self.lambda_L1)
        self.GDE = mod_to_gpu(self.GDE, self.gpu_id, self.device)
        self.criterionL2 = nn.MSELoss()
        self.criterionL1 = nn.L1Loss()

        if self.lambda_z > 0:
            RunningAverage(alpha=0.9, output_transform=lambda x: x['l1_z'].item()).attach(self.trainer, 'l1_z')
        if self.lambda_L1 > 0:
            RunningAverage(alpha=0.9, output_transform=lambda x: x['l1_Y'].item()).attach(self.trainer, 'l1_Y')

        RunningAverage(alpha=0.9, output_transform=lambda x: x['recon'].item()).attach(self.trainer, 'recon')

        if use_l1:
            self.recon_loss_func = self.criterionL1
        else:
            self.recon_loss_func = self.criterionL2

    def optimize(self, x, y, theta, label, **kwargs):

        self.y, self.x = y, x

        self.optim_gen.zero_grad()
        self.optim_dis.zero_grad()
        set_requires_grad(self.dis, True)
        set_requires_grad(self.gen, False)

        if self.fp16:
            self.x_a = self.x_a.half()

        for step_index in range(self.num_D_step):
            for accumulation_index in range(self.num_D_acc):
                self.z.sample_()

                pred_real, pred_fake = self.GDE(y=self.y, x=self.x, z=self.z, theta=theta, label=label,
                                                train_G=False)
                loss_dis_real, loss_dis_fake = self.gan_loss.D_loss(pred_fake, pred_real)
                loss_dis = (loss_dis_real + loss_dis_fake) / float(self.num_D_acc)
                loss_dis.backward()
            self.optim_dis.step()

        set_requires_grad(self.dis, False)
        set_requires_grad(self.gen, True)

        self.optim_gen.zero_grad()
        for accumulation_index in range(self.num_G_acc):
            self.z.sample_()

            pred_fake, pred_fake_recon, self.y_recon, z_hat, self.x_hat, self.y_tilde = self.GDE(y=self.y,
                                                                                                 x=None,
                                                                                                 z=self.z,
                                                                                                 theta=theta,
                                                                                                 label=label,
                                                                                                 train_G=True)
            loss_gen_gan = self.gan_loss.G_loss(pred_fake)
            if self.lmbda_gen_2 > 0:
                loss_gen_gan_recon = self.gan_loss.G_loss(pred_fake_recon)
            else:
                loss_gen_gan_recon = 0

            if self.lambda_L1 > 0:
                loss_gen_l1 = self.criterionL1(self.y_recon, self.y) * self.lambda_L1
            else:
                loss_gen_l1 = torch.tensor(0)
            if self.lambda_z > 0:
                loss_gen_z = self.criterionL1(z_hat, self.z) * self.lambda_z
            else:
                loss_gen_z = torch.tensor(0)

            loss_gen = (loss_gen_gan + loss_gen_gan_recon + loss_gen_l1 + loss_gen_z) / float(self.num_G_acc)
            loss_gen.backward()
        self.optim_gen.step()

        recon = self.criterionL2(self.x_hat, x)

        return dict(loss_gen=loss_gen,
                    l1_z= loss_gen_l1 / max(0.01, self.lambda_L1),
                    l1_Y= loss_gen_z / max(0.01, self.lambda_z),
                    loss_gen_l1=loss_gen_l1,
                    loss_dis_real=loss_dis_real,
                    recon=recon,
                    loss_dis_fake=loss_dis_fake)

    def write_image(self, engine, dataset_name):
        iteration = self.trainer.state.iteration

        b = engine.state.batch

        x = b['x'].cpu()
        y = b['y'].cpu()
        perm = torch.randperm(x.size(0))
        idx = perm[:8]
        list_tensor = []

        x = x[idx]
        y = y[idx]
        list_tensor.append(x.cpu())
        list_tensor.append(y.cpu())

        x_hat = self.x_hat[idx].cpu()
        list_tensor.append(x_hat)

        y_tilde = self.y_tilde[idx].cpu()
        list_tensor.append(y_tilde)
        if self.lambda_L1 > 0:
            y_recon = self.y_recon[idx].cpu()
            list_tensor.append(y_recon)
        img_tensor = torch.cat(list_tensor, dim=0)

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
                if self.nz > 0:
                    self.z.sample_()
                    z = self.z
                else:
                    z = None
                fake = self.gen(x=batch['y'].cuda(), z=z)
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
