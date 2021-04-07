#!/bin/bash
#SBATCH --job-name=test_bench
#SBATCH -A research
#SBATCH -c 35
#SBATCH -o bench_run.out
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

cd bench

python run_experiment.py name tf_para_reddit_no_bal path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 0 bal 0
python run_experiment.py name tf_para_reddit_bal_1 path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 0 bal 1
python run_experiment.py name tf_para_reddit_bal_2 path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 0 bal 2

python run_experiment.py name tf_para_reddit_no_bal path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 1 bal 0
python run_experiment.py name tf_para_reddit_bal_1 path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 1 bal 1
python run_experiment.py name tf_para_reddit_bal_2 path ../reddit test_mode 0 use_aug 1 aug_src para_aug.json model 1 bal 2

python run_experiment.py name reddit_no_bal path ../reddit test_mode 0 use_aug 0 model 0 bal 0
python run_experiment.py name reddit_bal_1 path ../reddit test_mode 0 use_aug 0 model 0 bal 1
python run_experiment.py name reddit_bal_2 path ../reddit test_mode 0 use_aug 0 model 0 bal 2

python run_experiment.py name reddit_no_bal path ../reddit test_mode 0 use_aug 0 model 1 bal 0
python run_experiment.py name reddit_bal_1 path ../reddit test_mode 0 use_aug 0 model 1 bal 1
python run_experiment.py name reddit_bal_2 path ../reddit test_mode 0 use_aug 0 model 1 bal 2

