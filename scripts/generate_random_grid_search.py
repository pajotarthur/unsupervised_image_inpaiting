import random

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open("configs/unsupervised_image_inpainting.yaml", "r") as stream:
    try:
        dico = yaml.load(stream, Loader=Loader)
    except yaml.YAMLError as exc:
        print(exc)

dico['corruption']['percent'] = 0.20


x = [[True, False], [False, True], [False, False]]

c = random.choice(x)
dico["modules"]['dis']['_name'] = 'biggan_dis' #random.choice([64, 32])
dico["modules"]['dis']['ndf'] = random.choice([32])
dico["modules"]['dis']['spectral'] = c[0] #random.choice([True, False])
dico["modules"]['dis']['wheigth'] = c[1] # random.choice([True, False])
dico["modules"]['dis']['D_attn'] = random.choice(['128'])
dico["modules"]['dis']['activation'] = random.choice(['relu'])
dico["modules"]['dis']['init_gain'] = random.choice([1.41])
dico["modules"]['dis']['init_name'] = random.choice(['orthogonal'])


x = [[True, False], [False, True], [False, False]]

c = random.choice(x)
dico["modules"]['enc']['_name'] = 'biggan_dis' #random.choice([64, 32])
dico["modules"]['enc']['ndf'] = random.choice([32])
dico["modules"]['enc']['spectral'] = c[0] #random.choice([True, False])
dico["modules"]['enc']['wheigth'] = c[1] # random.choice([True, False])
dico["modules"]['enc']['D_attn'] = random.choice(['128'])
dico["modules"]['enc']['activation'] = random.choice(['relu'])
dico["modules"]['enc']['init_gain'] = random.choice([1.41])
dico["modules"]['enc']['init_name'] = random.choice(['orthogonal'])

c = random.choice(x)

dico["modules"]['gen']['_name'] = 'pix2pix' #random.choice([64, 32])
dico["modules"]['gen']['ngf'] = random.choice([32])



expe_name = 'unsup'

dico["experiment"]['num_dis_step'] = random.choice([1])
dico["experiment"]['type_gan'] = random.choice(['hinge', 'lsgan'])
dico["experiment"]['truncation'] = random.choice([0])
dico["experiment"]['lambda_L1'] = random.choice([0, 1, 5, 10, 50])
dico["experiment"]['lambda_z'] = random.choice([0, 1, 5, 10, 50])
dico["experiment"]['nz'] = random.choice([16, 32])

dico["experiment"]['fid'] = True
dico["experiment"]['root'] = '/net/cluster/pajot/logs/unir/' + expe_name
dico["experiment"]['nepoch'] = 50
dico["experiment"]['nz'] = dico["modules"]['gen']['nz']


# dico["optimizers"]['optim_dis']['_name'] = 'adam16'#random.choice([0])
dico["optimizers"]['optim_dis']['betas'][0] = random.choice([0])
dico["optimizers"]['optim_dis']['lr'] = random.choice([4e-4])

# dico["optimizers"]['optim_gen']['_name']= 'adam16'#random.choice([0])
dico["optimizers"]['optim_gen']['betas'][0] = random.choice([0])
dico["optimizers"]['optim_gen']['lr'] = random.choice([1e-4])

with open("configs/grid_search/" + expe_name + ".yaml", 'w') as stream:
    try:
        dico = yaml.dump(dico, stream)
    except yaml.YAMLError as exc:
        print(exc)
