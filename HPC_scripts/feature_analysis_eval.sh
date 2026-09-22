#!/bin/bash
#SBATCH --job-name=Feature-Analysis
#SBATCH --partition=GPU,INFERENCE
#SBATCH --mem=24G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --error=/home/ctheod03/repo/BranchInformations/logs/%x_%j.err

source /home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate env

export PYTHONPATH=/home/ctheod03/repo/BranchInformations:/home/ctheod03/repo/BranchInformations/mlp_predictor

cd /home/ctheod03/repo/BranchInformations/mlp_predictor

mkdir -p ./tmp_fa_${SLURM_JOB_ID}

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group none > ./tmp_fa_${SLURM_JOB_ID}/none.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group opcode_identity > ./tmp_fa_${SLURM_JOB_ID}/opcode_identity.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group register_identity > ./tmp_fa_${SLURM_JOB_ID}/register_identity.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group surrounding_context > ./tmp_fa_${SLURM_JOB_ID}/surrounding_context.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group taken_path_structure > ./tmp_fa_${SLURM_JOB_ID}/taken_path_structure.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group fall_through_structure > ./tmp_fa_${SLURM_JOB_ID}/fall_through_structure.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group loop_locality > ./tmp_fa_${SLURM_JOB_ID}/loop_locality.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group memory_behavior > ./tmp_fa_${SLURM_JOB_ID}/memory_behavior.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group routine_context > ./tmp_fa_${SLURM_JOB_ID}/routine_context.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --group code_position > ./tmp_fa_${SLURM_JOB_ID}/code_position.out &

wait

cat ./tmp_fa_${SLURM_JOB_ID}/*.out > feature_analysis_all_groups.out
rm -rf ./tmp_fa_${SLURM_JOB_ID}
