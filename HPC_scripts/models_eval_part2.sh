#!/bin/bash
#SBATCH --job-name=Models-Eval-2
#SBATCH --partition=GPU
#SBATCH --mem=58G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --error=/home/ctheod03/repo/BranchInformations/logs/%x_%j.err

source /home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate env


export PYTHONPATH=/home/ctheod03/repo/BranchInformations:/home/ctheod03/repo/BranchInformations/mlp_predictor

cd /home/ctheod03/repo/BranchInformations/mlp_predictor

mkdir -p ./tmp4

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "526.blender_r" --min-executed 50 > ./tmp4/espblender.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "526.blender_r" --min-executed 50 > ./tmp4/ssblender.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "526.blender_r" --min-executed 50 > ./tmp4/embblender.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "527.cam4_r" --min-executed 50 > ./tmp4/espcam4.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "527.cam4_r" --min-executed 50 > ./tmp4/sscam4.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "527.cam4_r" --min-executed 50 > ./tmp4/embcam4.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "531.deepsjeng_r" --min-executed 50 > ./tmp4/espdeepsjeng.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "531.deepsjeng_r" --min-executed 50 > ./tmp4/ssdeepsjeng.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "531.deepsjeng_r" --min-executed 50 > ./tmp4/embdeepsjeng.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "538.imagick_r" --min-executed 50 > ./tmp4/espimagick.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "538.imagick_r" --min-executed 50 > ./tmp4/ssimagick.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "538.imagick_r" --min-executed 50 > ./tmp4/embimagick.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "541.leela_r" --min-executed 50 > ./tmp4/espleela.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "541.leela_r" --min-executed 50 > ./tmp4/ssleela.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "541.leela_r" --min-executed 50 > ./tmp4/embleela.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "544.nab_r" --min-executed 50 > ./tmp4/espnab.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "544.nab_r" --min-executed 50 > ./tmp4/ssnab.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "544.nab_r" --min-executed 50 > ./tmp4/embnab.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "548.exchange2_r" --min-executed 50 > ./tmp4/espexchange2.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "548.exchange2_r" --min-executed 50 > ./tmp4/ssexchange2.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "548.exchange2_r" --min-executed 50 > ./tmp4/embexchange2.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "549.fotonik3d_r" --min-executed 50 > ./tmp4/espfotonik3d.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "549.fotonik3d_r" --min-executed 50 > ./tmp4/ssfotonik3d.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "549.fotonik3d_r" --min-executed 50 > ./tmp4/embfotonik3d.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "554.roms_r" --min-executed 50 > ./tmp4/esproms.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "554.roms_r" --min-executed 50 > ./tmp4/ssroms.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "554.roms_r" --min-executed 50 > ./tmp4/embroms.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "557.xz_r" --min-executed 50 > ./tmp4/espxz.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "557.xz_r" --min-executed 50 > ./tmp4/ssxz.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "557.xz_r" --min-executed 50 > ./tmp4/embxz.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "503.bwaves_r" --min-executed 50 > ./tmp4/espbwaves.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "503.bwaves_r" --min-executed 50 > ./tmp4/ssbwaves.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "503.bwaves_r" --min-executed 50 > ./tmp4/embbwaves.out &

wait

# Combine all the temporary files into the final file
cat ./tmp4/*.out > smr_n2_exe50.out

# Delete the temporary files so your folder stays clean
rm ./tmp4/*.out
