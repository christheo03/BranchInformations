#!/bin/bash
#SBATCH --job-name=Single-Bench-Eval
#SBATCH --partition=GPU
#SBATCH --nodelist=gpu01
#SBATCH --mem=24G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --error=/home/ctheod03/repo/BranchInformations/logs/%x_%j.err

source /home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate env

export PYTHONPATH=/home/ctheod03/repo/BranchInformations:/home/ctheod03/repo/BranchInformations/mlp_predictor

cd /home/ctheod03/repo/BranchInformations/mlp_predictor

BENCH="$1"
MIN_EXECUTED="${2:-0}"

mkdir -p ./tmp

python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "$BENCH" --min-executed "$MIN_EXECUTED" > ./tmp/esp_${BENCH}.out &
python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "$BENCH" --min-executed "$MIN_EXECUTED" > ./tmp/ss_${BENCH}.out &
python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "$BENCH" --min-executed "$MIN_EXECUTED" > ./tmp/emb_${BENCH}.out &

wait

cat ./tmp/esp_${BENCH}.out ./tmp/ss_${BENCH}.out ./tmp/emb_${BENCH}.out > evaluation_${BENCH}.out
