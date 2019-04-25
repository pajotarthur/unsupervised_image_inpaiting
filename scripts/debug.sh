#!/bin/sh

name="$1"
device="$2"
config_updates="${@:3}"

loglevel="31"

options="
    --name $name
    --loglevel $loglevel
    --unobserved
    --print_config
    "

config="configs/$name.yaml"

run="
    find -name "*.pyc" -delete &&
    CUDA_VISIBLE_DEVICES=$device
    /local/pajot/anaconda/envs/pytorch/bin/python run.py $options with $config ${@:3} experiment.root=None experiment.gpu_id=$device
    "

printf "$run \n"
eval $run