#!/bin/sh

name="$1"
device="$2"
config_updates="${@:3}"

loglevel="60"
project_name="unir"
mongo_db="drunk:27017:$project_name.runs"

options="
    --name $name
    --loglevel $loglevel
    --mongo_db $mongo_db
    --print_config
    "

config="configs/$name.yaml"

cleanup ()
{
kill -s SIGTERM $!
exit 0
}

trap cleanup SIGINT SIGTERM

while [[ 1 ]]
do
    run="
    find . -name "*.pyc" -delete &&
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$device
    /local/pajot/anaconda/envs/pytorch/bin/python run.py $options with $config ${@:3}  experiment.gpu_id=$device
    "
    printf "$run \n"
    /local/pajot/anaconda/envs/pytorch/bin/python scripts/generate_random_grid_search.py
    eval $run
    wait $!
done