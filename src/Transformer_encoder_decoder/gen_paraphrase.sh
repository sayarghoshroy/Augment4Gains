#!/bin/bash
#SBATCH --job-name=reddit_gen_para
#SBATCH -A research
#SBATCH -c 18
#SBATCH -o reddit_gen.out
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

python run_paraphraser.py

