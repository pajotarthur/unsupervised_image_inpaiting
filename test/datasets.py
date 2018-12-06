import yaml

from src.datasets import init_dataset


configs = yaml.load(open('default_configs/datasets.yaml'))

datasets_names = ['mnist', 'fashion_mnist']
for name in datasets_names:
    dl = init_dataset(name, **configs[name])

    print('dataset: {}'.format(name))
    for batch in dl:
        for key, value in batch.items():
            print(key, value.shape)
        break
    print()