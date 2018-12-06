import yaml
from pprint import pprint

config = yaml.load(open('configs/mnist.yaml'))

pprint(config)