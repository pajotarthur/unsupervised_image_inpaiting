from .GAN_experiment import GAN_experiment
from src.utils.net import set_requires_grad
from torch import nn
import torch
from torchvision.utils import make_grid


# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
    def __init__(self, G, D, corruption):
        super(G_D, self).__init__()
        self.G = G
        self.D = D
        self.corruption = corruption

    def forward(self, z, y=None, label=None, train_G=False, return_G_z=False):
        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            x_hat = self.G(z, label)
            #only if ambientgan styple
            y_hat, _ = self.corruption(x_hat, device=x_hat.device)

        # if y is not None:
        # print(f'y_min {y.max().item()} {y.min().item()} {y.mean().item()}')
        # print(f'x {y_hat.max().item()} {y_hat.min().item()} {y_hat.mean().item()}')
        # print(f'x_hat {x_hat.max().item()} {x_hat.min().item()} {x_hat.mean().item()}')

        if train_G:
            pred_fake = self.D(y_hat, label)
            return pred_fake , x_hat, y_hat
        else:
            pred_real = self.D(y, label)
            pred_fake = self.D(y_hat, label)

            return pred_real, pred_fake


class AmbientGan(GAN_experiment):
    def __init__(self, **kwargs):
        super(AmbientGan, self).__init__(**kwargs)
        self.GD = G_D(self.gen, self.dis, self.corruption)


    def optimize(self, x, y, label, **kwargs):

        self.optim_gen.zero_grad()
        self.optim_dis.zero_grad()
        set_requires_grad(self.dis, True)
        set_requires_grad(self.gen, False)

        if self.fp16:
            y = y.half()

        for step_index in range(self.num_D_step):
            for accumulation_index in range(self.num_D_acc):
                self.z.sample_()

                pred_real, pred_fake = self.GD(z=self.z, y=y, label=label,
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
            pred_fake, self.x_hat, self.y_hat = self.GD(z=self.z, y=None, label=label, train_G=True)
            loss_gen = self.gan_loss.G_loss(pred_fake)
            loss_gen = loss_gen / float(self.num_G_acc)
            loss_gen.backward()
        self.optim_gen.step()

        return dict(loss_gen=loss_gen, loss_dis_real=loss_dis_real, loss_dis_fake=loss_dis_fake)

    def write_image(self, engine, dataset_name):
        iteration = self.trainer.state.iteration

        b = engine.state.batch

        x = b['x'].cpu()
        y = b['y'].cpu()
        perm = torch.randperm(x.size(0))
        idx = perm[:8]
        x = x[idx]
        y = y[idx]

        x_hat = self.x_hat[:self.num_samples]
        y_hat = self.y_hat[:self.num_samples]
        # sample = self.samples(self.num_samples)
        # interp_tensor = self.interp(self.num_samples, fix=False)
        img_tensor = torch.cat([x.cpu(), y.cpu(), x_hat.cpu(), y_hat.cpu()], dim=0)


        # img_tensor = torch.cat([x.cpu(), y.cpu(), sample.cpu(), interp_tensor.cpu()], dim=0)

        img = make_grid(
                img_tensor,
                nrow=self.num_samples, scale_each=True, normalize=True,
                )

        try:
            self.writers.add_image(dataset_name, img, iteration)
        except:
            self.pbar.log_message('IMPOSSIBLE TO SAVE')