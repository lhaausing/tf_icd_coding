#!/bin/bash

#SBATCH --partition=gpu8_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:8

model_name=bert-base-uncased
batch_size=64
n_gpu=8
n_epochs=40
checkpt_path=../local_${model_name}_bs${batch_size}_sepcls.pt

module load cuda91/toolkit/9.1.85
python3 main.py \
  --data_dir ../data \
  --model_name ../${model_name} \
  --local_model \
  --n_epochs ${n_epochs} \
  --batch_size ${batch_size} \
  --checkpt_path ${checkpt_path} "
