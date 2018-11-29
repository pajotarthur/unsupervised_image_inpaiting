import os
import pprint
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import tqdm
from sacred import Experiment
from tqdm import tqdm

import dlda.modules.losses as losses
import factory.closure as closure_factory
import factory.corruption as corruption_factory
import factory.dataset as dataset_factory
import factory.lr_scheduler as lr_scheduler_factory
import factory.model as model_factory
import factory.optimizer as optimizer_factory
import utils.external_resources as external
from utils.meter import AverageMeter, ScalarMeter
from utils.utils import CustomInterrupt

ex = Experiment("sagan_new_params")

# setting up experiment folder structure
time_str = datetime.now().strftime("%a-%b-%d-%H:%M:%S")
exp_dir = os.path.join('/net/drunk/debezenac/expes', ex.path, time_str)
os.makedirs(exp_dir)
print('Experiment folder: {}'.format(exp_dir))

ex.observers.append(external.get_mongo_obs(debug=False))

@ex.config
def config_exp():
    device = 'cuda:0'
    nepochs = 600
    niter_train = 300
    niter_test = 30
    min_start_mse = 2.
    two_step_dis = False


create_optim = optimizer_factory.optim_with_encoder(ex)
create_scheduler = lr_scheduler_factory.lr_scheduler(ex)
create_dataset = dataset_factory.dataset(ex)
create_model = model_factory.model_with_encoder(ex)
create_corruption = corruption_factory.corruption(ex)
create_closure = closure_factory.closure(ex)


@ex.main
def main(_run, _seed, device, nepochs, niter_train, niter_test, min_start_mse, corruption, gen, closure, two_step_dis):
    pprint.pprint(_run.config)

    assert(gen['nz'] == closure['nz'])

    torch.cuda.manual_seed_all(_seed)
    torch.manual_seed(_seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    np.random.seed(_seed)
    dict_var = {}
    _run.info['exp_dir'] = exp_dir
    corrupt = create_corruption()

    data, nc = create_dataset(corruption=corrupt)
    dict_res = create_model(nc=nc)
    dict_optim = create_optim(dict_res=dict_res)
    dict_scheduler = create_scheduler(dict_optim=dict_optim)

    dict_criterion = {
        'loss_GAN': losses.GANLoss().to(device),
        'loss_GANEnt': losses.GANLoss().to(device),
        'loss_L1_measurements': torch.nn.L1Loss().to(device),
        'loss_L1': torch.nn.L1Loss().to(device),
        'loss_L2': torch.nn.MSELoss().to(device),
        'loss_z': losses.MMDLoss().to(device),
    }


    closure = create_closure(dict_criterion=dict_criterion, dict_optim=dict_optim, dict_res=dict_res, )

    meters = {
        'loss_dis': AverageMeter(),
        'loss_gen': AverageMeter(),
        'loss_MSE': AverageMeter(),
        'loss_z': AverageMeter(),
    }

    dict_criterion['loss_masked_MSE'] = losses.MaskedMSELoss().to(device)
    meters['loss_masked_MSE'] = AverageMeter()
    dict_criterion['loss_masked_std'] = losses.MaskedVarLoss(gen['nz']).to(device)
    meters['loss_masked_std'] = AverageMeter()

    meters.update({'lr_' + k: ScalarMeter() for k, v in dict_scheduler.items()})
    meters.update(closure.__dict__.get('meters', {}))

    min_avg_loss = np.inf
    bar_epoch = tqdm(range(1, nepochs + 1), ncols=0)
    for epoch in bar_epoch:
        for split, dl in data.items():
            for m in meters.values():
                m.reset()
            with torch.set_grad_enabled(split == 'train'):
                iter = 0
                niter = (split == 'test') and niter_test or niter_train
                bar = tqdm(dl, ncols=0, total=min(niter, len(dl)))

                if split == 'train': closure.scheduler_step()

                for dict_var in bar:
                    iter += 1
                    for var_name, var in dict_var.items():
                        dict_var[var_name] = var.to(device)

                    dict_var = closure.forward(corrupt, device, dict_var)
                    if split == 'train':
                        loss_D = closure.backward_D(dict_var)

                        if two_step_dis and (iter % 2 == 0):
                            continue
                            
                        loss_G = closure.backward_G(dict_var)

                    loss_state = dict_criterion['loss_L2'](dict_var['fake_sample'], dict_var['sample'])
                    meters['loss_MSE'].update(loss_state.item(), dict_var['fake_sample'].size(0))

                    if corruption['name'] in ['keep_patch', 'remove_pix_dark', 'remove_pix']:
                        if corruption['name'] == 'keep_patch':
                            val = 255
                        else:
                            val = 1
                        mask_mse = 1 - (dict_var['theta'].squeeze(1) == val).float()
                        masked_mse = dict_criterion['loss_masked_MSE'](dict_var['fake_sample'], dict_var['sample'], mask_mse)
                        meters['loss_masked_MSE'].update(masked_mse.item(), dict_var['fake_sample'].size(0))

                        if split != "train":
                            if gen['nz'] > 0:
                                std = dict_criterion['loss_masked_std'](dict_res['netG'], dict_var['fake_sample'],
                                                                        (1 - mask_mse).float())
                            else:
                                std = torch.zeros(1)
                            meters['loss_masked_std'].update(std.item(), dict_var['fake_sample'].size(0))

                    else:
                        meters['loss_masked_MSE'].update(0, dict_var['fake_sample'].size(0))

                    if iter % 1 == 0:
                        if split == 'train':
                            meters['loss_dis'].update(loss_D.item(), dict_var['fake_sample'].size(0))
                            meters['loss_gen'].update(loss_G.item(), dict_var['fake_sample'].size(0))
                            bar.set_postfix_str(
                                "{}: {} G:{:.4f} D={:.4f} MM={:.4f} M={:.4f}".format(_run._id, epoch,
                                                                                     meters['loss_gen'].get(),
                                                                                     meters['loss_dis'].get(),
                                                                                     meters['loss_masked_MSE'].get(),
                                                                                     meters['loss_MSE'].get(),
                                                                                     ))
                        else:
                            bar.set_description_str(
                                "TEST {}: {} MV={:.6f} MM={:.6f} M={:.6f}".format(_run._id, epoch,
                                                                                  meters['loss_masked_std'].get(),
                                                                                  meters['loss_masked_MSE'].get(),
                                                                                  meters['loss_MSE'].get()))

                    if iter > 15 and epoch < 10 and meters['loss_MSE'].get() > min_start_mse:
                        print('Stopped start MSE to big: {} > {}'.format(meters['loss_MSE'].get(), min_start_mse))
                        _run.result = meters['loss_MSE'].get()
                        raise CustomInterrupt("START_RESULT_TOO_SMALL")

                    if iter > niter:
                        break
            try:
                if _run.config['closure']['name'] == "superres":
                    cat = torch.cat([v for k, v in dict_var.items() if v.shape[1] == nc and len(v.shape) == 4 and k != 'mask' if v.shape[2]==128])
                    cat_lowres = torch.cat([v for k, v in dict_var.items() if v.shape[1] == nc and len(v.shape) == 4 and k != 'mask' and v.shape[2]!=128])
                    vutils.save_image(cat_lowres, exp_dir + '/' + split + 'lowres_' + str(epoch) + '.png', scale_each=True,
                      normalize=True, nrow=_run.config['dataset']['batch_size'])
                    vutils.save_image(cat, exp_dir + '/' + split + '_' + str(epoch) + '.png', scale_each=True,
                                      normalize=True, nrow=_run.config['dataset']['batch_size'])
                else:
                    cat = torch.cat([v for k, v in dict_var.items() if v.shape[1] == nc and len(v.shape) == 4 and k != 'mask'])
            except:
                pass

            if iter % 1:
                bar_epoch.set_description_str("id: {} G_gan={:.4f} D={:.4f} z={:.4f} MV={:.4f} MM={:.4f} M={:.4f}".format(_run._id,
                                                                                        meters['loss_gen'].get(),
                                                                                        meters['loss_dis'].get(),
                                                                                        meters['loss_z'].get(),
                                                                                        meters['loss_masked_std'].get(),    
                                                                                        meters['loss_masked_MSE'].get(),    
                                                                                        meters['loss_MSE'].get()))
            if split == 'train':

                if meters['loss_dis'].get() < 0.001:
                    print('Stopped because Discriminator loss: {} < 0.001:'.format(meters['loss_dis'].get()))
                    _run.result = meters['loss_MSE'].get()
                    raise CustomInterrupt('DIS_TOO_SMALL')

                for k, v in dict_scheduler.items():
                    meters['lr_' + k].update(v.get_lr()[0])
                    v.step()

            if split == 'test':

                def save_checkpoint(dict_res, min_avg_loss, epoch, filename='best_model.pth.tar'):
                    dict_save = {
                        'epoch': epoch,
                        'generator': dict_res['netG'],
                        'min_avg_loss': min_avg_loss,
                    }
                    if 'netE' in dict_res:
                        dict_save['encoder'] = dict_res["netE"]

                    path = os.path.join(exp_dir, filename)
                    print('saving to {}...'.format(path))
                    torch.save(dict_save, path)

                if (epoch % 100 == 0):
                    if epoch > 20:
                        save_checkpoint(dict_res, meters['loss_MSE'].get(), epoch,
                                        filename="model_" + str(epoch) + "pth.tar")

                min_avg_loss = min(min_avg_loss, meters['loss_MSE'].get())
                _run.result = min_avg_loss

            for name, m in meters.items():
                tag = 'meters' + '/' + name + '/' + split
                ex.log_scalar(tag, m.get(), epoch)

    return min_avg_loss


if __name__ == '__main__':
    ex.run_commandline()
