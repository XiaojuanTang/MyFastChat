#! /bin/bash

#SBATCH -J finetune
#SBATCH -p gpu
#SBATCH -c 128
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --account=research
#SBATCH --qos=level0
#SBATCH -N 1
#SBATCH --gres=gpu:4

bash train_llama2_7b.sh