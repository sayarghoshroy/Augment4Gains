#!/bin/bash
#SBATCH --job-name=test_bench
#SBATCH -A research
#SBATCH -c 35
#SBATCH -o bench_try.out
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

cd bench
python run_experiment.py name reddit_bert_bal path ../reddit test_mode 0 use_aug 0 model 0 bal 1
python run_experiment.py name reddit_bert_nobal path ../reddit test_mode 0 use_aug 0 model 0 bal 0
python run_experiment.py name reddit_roberta_bal path ../reddit test_mode 0 use_aug 0 model 1 bal 1
python run_experiment.py name reddit_roberta_nobal path ../reddit test_mode 0 use_aug 0 model 1 bal 0


