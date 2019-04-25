import torch
from ignite.metrics import RunningAverage
from scipy.stats import truncnorm
from torch import nn
from torchvision.utils import make_grid

import src.utils.fid.fid as metrics
from src.experiments.dataset_experiment import DatasetExperiment
from src.modules.GD_factory import G_D
from src.modules.losses import GANLoss
from src.utils.fid.inception import InceptionV3
from src.utils.net import Distribution
from src.utils.net import set_requires_grad


class GAN_experiment(DatasetExperiment):
    def __init__(self, gen, dis, optim_gen, optim_dis, nz, random_type='normal', fid=True, num_samples=8, truncation=0,
                 fp16=False, G_batch_size=16, num_D_step=1, num_D_acc=1, num_G_acc=1, gan_mode='hinge',
                 **kwargs):
        super(GAN_experiment, self).__init__(**kwargs)

        self.nz = nz
        self.random_type = random_type
        self.num_D_step = num_D_step
        self.num_D_acc = num_D_acc
        self.num_G_acc = num_G_acc

        self.truncation = truncation

        self.num_samples = num_samples

        self.gen = gen.to(self.device)
        self.dis = dis.to(self.device)

        self.optim_gen = optim_gen
        self.optim_dis = optim_dis

        self.fp16 = fp16

        RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_gen'].item()).attach(self.trainer, 'gen')
        RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_dis_real'].item()).attach(self.trainer, 'dis_real')
        RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_dis_fake'].item()).attach(self.trainer, 'dis_fake')

        # self.pbar.attach(self.trainer, metric_names=['gen', 'dis_real', 'dis_fake'])
        self.pbar.attach(self.trainer, metric_names=[])

        G_batch_size = self.train.batch_size
        self.z = Distribution(torch.randn(G_batch_size, self.nz, requires_grad=False))
        self.z.init_distribution(dist_type=random_type, mean=0, var=1)
        self.z = self.z.to(self.device, torch.float16 if fp16 else torch.float32)
        if fp16:
            self.z = self.z.half()

        if self.fp16:
            self.gen = self.gen.half()
            self.dis = self.dis.half()

        self.GD = G_D(self.gen, self.dis)
        self.gan_loss = GANLoss(gan_mode=gan_mode)

        if type(self.gpu_id) == int:
            self.gpu_id = [self.gpu_id]
        if len(self.gpu_id) > 1:
            assert (torch.cuda.is_available())
            self.GD = torch.nn.DataParallel(self.GD, self.gpu_id).cuda()  # multi-GPUs
        else:
            self.GD = self.GD.to(self.device)

        self.fid = fid
        if self.fid:
            self.max_fid = float('inf')

            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            self.model = InceptionV3([block_idx])
            self.fid_score = float('inf')
            if len(self.gpu_id) > 1:
                assert (torch.cuda.is_available())
                self.model = torch.nn.DataParallel(self.model, self.gpu_id).cuda()  # multi-GPUs
            else:
                self.model = self.model.to(self.device)



    def optimize(self, x, label, **kwargs):
        self.optim_gen.zero_grad()
        self.optim_dis.zero_grad()

        set_requires_grad(self.dis, True)
        set_requires_grad(self.gen, False)

        if self.fp16:
            x = x.half()

        for step_index in range(self.num_D_step):
            for accumulation_index in range(self.num_D_acc):
                self.z.sample_()

                pred_real, pred_fake = self.GD(z=self.z, x=x, label=label, train_G=False)
                loss_dis_real, loss_dis_fake = self.gan_loss.D_loss(pred_fake, pred_real)
                loss_dis = (loss_dis_real + loss_dis_fake) / float(self.num_D_acc)
                loss_dis.backward()
            self.optim_dis.step()

        set_requires_grad(self.dis, False)
        set_requires_grad(self.gen, True)

        self.optim_gen.zero_grad()
        for accumulation_index in range(self.num_G_acc):
            self.z.sample_()
            pred_fake = self.GD(z=self.z, x=None, label=label, train_G=True)
            loss_gen = self.gan_loss.G_loss(pred_fake)
            loss_gen = loss_gen / float(self.num_G_acc)
            loss_gen.backward()
        self.optim_gen.step()

        return dict(loss_gen=loss_gen, loss_dis_real=loss_dis_real, loss_dis_fake=loss_dis_fake)

    def K_step(self, engine):
        i = self.trainer.state.iteration
        self.compute_fid(i)
        self.write_image(engine, 'train')

    def write_image(self, engine, dataset_name):
        iteration = self.trainer.state.iteration

        sample = self.samples(self.num_samples)
        interp_tensor = self.interp(self.num_samples, fix=False)


        img_tensor = torch.cat([sample, interp_tensor], dim=0)

        img = make_grid(
                img_tensor,
                nrow=self.num_samples, scale_each=True, normalize=True,
                )

        try:
            self.writers.add_image(dataset_name, img, iteration)
        except:
            self.pbar.log_message('IMPOSSIBLE TO SAVE')

    def samples(self, num_samples=12):
        ims = []

        z_dist = Distribution(torch.randn(num_samples, self.nz, requires_grad=False))
        z_dist.init_distribution(dist_type=self.random_type, mean=0, var=1)
        z_dist = z_dist.to(self.device, torch.float16 if self.fp16 else torch.float32)
        if self.fp16:
            z_dist = z_dist.half()

        # for j in range(sample_per_sheet):
        # batch = convert_tensor(batch, self.device)
        z_dist.sample_()
        with torch.no_grad():
            if self.truncation > 0:
                z = self.truncated_z_sample(batch_size=num_samples, truncation=self.truncation).to(self.device)
                o = self.gen(z=z)
            else:
                self.z.sample_()
                o = self.gen(z=z_dist)

            if self.fp16:
                o = o.float()

        out_ims = o
        return o

    def interp(self, num_samples=12, fix=True):

        def interpolation(x0, x1, num_midpoints):
            lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cuda').to(x0.dtype)
            return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))

        if self.truncation > 0:
            zs = interpolation(self.truncated_z_sample(batch_size=1, truncation=self.truncation).to(self.device),
                               self.truncated_z_sample(batch_size=1, truncation=self.truncation).to(self.device),
                               num_samples - 2).view(-1, self.nz)

        else:
            zs = interpolation(torch.randn(1, 1, self.nz, device=self.device),
                               torch.randn(1, 1, self.nz, device=self.device),
                               num_samples - 2).view(-1, self.nz)

        with torch.no_grad():
            if self.fp16:
                zs = zs.half()
            # if self.truncation > 0:
            o = self.gen(z=zs)
            if self.fp16:
                o = o.float()

        out_ims = o
        return o

    def compute_fid(self, iteration):
        self.evaluating()
        fake_list, real_list = [], []
        with torch.no_grad():
            for i, batch in enumerate(self.eval):
                if self.truncation > 0:
                    z = self.truncated_z_sample(batch_size=batch['x'].size(0), truncation=self.truncation).to(self.device)
                    fake = self.gen(z=z)
                else:
                    self.z.sample_()
                    fake = self.gen(z=self.z)
                true = batch['x']
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

    def extra_log(self):
        log = ''
        log += f" FID: {self.fid_score:.6f}"
        log += ' Time {:.3f}[s] done. Time per batch: {:.3f}[s]'.format(self.timer.total, self.timer.value())
        return log

    def truncated_z_sample(self, batch_size, truncation=1., seed=None):
        values = truncnorm.rvs(-2, 2, size=(batch_size, self.nz))
        return torch.tensor(truncation * values).float()
