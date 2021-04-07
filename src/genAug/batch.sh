#!/bin/bash
#SBATCH -n 16
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=40
#SBATCH --gres=gpu:4
#SBATCH --mail-user=souvik.banerjee@research.iiit.ac.in
#SBATCH --mail-type=ALL
module add cuda/8.0
module add cudnn/7-cuda-8.0

python3 genAug.py --label 1
