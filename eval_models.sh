#!/bin/bash
#SBATCH --job-name=esp_mcf
#SBATCH --partition=GPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --error=/home/ctheod03/BranchInformations/logs/%x_%j.err

source /home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate env

export PYTHONPATH=/home/ctheod03/BranchInformations:/home/ctheod03/BranchInformations/mlp_predictor

cd /home/ctheod03/BranchInformations/mlp_predictor
python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "505.mcf_r" >> evaluation.out &
python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "505.mcf_r" >> evaluation.out &
python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "505.mcf_r" >> evaluation.out &
wait
