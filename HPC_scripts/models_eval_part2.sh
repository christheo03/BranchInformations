#!/bin/bash
#SBATCH --job-name=Models-Eval-2
#SBATCH --partition=GPU,INFERENCE
#SBATCH --mem=20G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --error=/home/ctheod03/repo/BranchInformations/logs/%x_%j.err

source /home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate env


export PYTHONPATH=/home/ctheod03/repo/BranchInformations:/home/ctheod03/repo/BranchInformations/mlp_predictor

cd /home/ctheod03/repo/BranchInformations/mlp_predictor

mkdir -p ./tmp2

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "526.blender_r" --min-executed 1 > ./tmp2/embblender.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "527.cam4_r" --min-executed 1 > ./tmp2/embcam4.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "531.deepsjeng_r" --min-executed 1 > ./tmp2/embdeepsjeng.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "538.imagick_r" --min-executed 1 > ./tmp2/embimagick.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "541.leela_r" --min-executed 1 > ./tmp2/embleela.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "544.nab_r" --min-executed 1 > ./tmp2/embnab.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "548.exchange2_r" --min-executed 1 > ./tmp2/embexchange2.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "549.fotonik3d_r" --min-executed 1 > ./tmp2/embfotonik3d.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "554.roms_r" --min-executed 1 > ./tmp2/embroms.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "557.xz_r" --min-executed 1 > ./tmp2/embxz.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "503.bwaves_r" --min-executed 1 > ./tmp2/embbwaves.out &

wait

# Combine all the temporary files into the final file
cat ./tmp2/*.out > wmr_n2_exe3.out

# Delete the temporary files so your folder stays clean
rm ./tmp2/*.out
