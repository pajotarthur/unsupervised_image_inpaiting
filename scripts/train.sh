#!/bin/sh

name="$1"
device="$2"
config_updates="${@:3}"

loglevel="40"
project_name="unir"
mongo_db="drunk:27017:$project_name.runs"

options="
    --name $name
    --loglevel $loglevel
    --mongo_db $mongo_db
    --print_config
    "

config="configs/$name.yaml"

run="
    find -name "*.pyc" -delete &&
    CUDA_VISIBLE_DEVICES=$device
    /local/pajot/anaconda/envs/pytorch/bin/python run.py $options with $config ${@:3}
    "

printf "$run \n"
eval $run