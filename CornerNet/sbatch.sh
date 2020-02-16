#!/bin/sh
#SBATCH -J JK
#SBATCH --time=24:00:00
#SBATCH -p skl_v100nv_4
#SBATCH --comment pytorch
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -o result/%x_%j.out
#SBATCH -e result/%x_%j.err
#SBATCH --gres=gpu:4

export PATH=$PATH:/home01/q542a02/.local/bin

python train.py -c config.json

