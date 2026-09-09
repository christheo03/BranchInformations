#!/bin/bash
#SBATCH --job-name=Feature-Analysis
#SBATCH --partition=GPU
#SBATCH --mem=56G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --error=/home/ctheod03/repo/BranchInformations/logs/%x_%j.err

source /home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate env

export PYTHONPATH=/home/ctheod03/repo/BranchInformations:/home/ctheod03/repo/BranchInformations/mlp_predictor

cd /home/ctheod03/repo/BranchInformations/mlp_predictor

GROUP="$1"

mkdir -p ./tmp_fa

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "500.perlbench_r" --group "$GROUP" > ./tmp_fa/${GROUP}_perlbench.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "502.gcc_r" --group "$GROUP" > ./tmp_fa/${GROUP}_gcc.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "503.bwaves_r" --group "$GROUP" > ./tmp_fa/${GROUP}_bwaves.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "505.mcf_r" --group "$GROUP" > ./tmp_fa/${GROUP}_mcf.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "507.cactuBSSN_r" --group "$GROUP" > ./tmp_fa/${GROUP}_cactu.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "508.namd_r" --group "$GROUP" > ./tmp_fa/${GROUP}_namd.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "510.parest_r" --group "$GROUP" > ./tmp_fa/${GROUP}_parest.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "511.povray_r" --group "$GROUP" > ./tmp_fa/${GROUP}_povray.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "519.lbm_r" --group "$GROUP" > ./tmp_fa/${GROUP}_lbm.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "520.omnetpp_r" --group "$GROUP" > ./tmp_fa/${GROUP}_omnet.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "521.wrf_r" --group "$GROUP" > ./tmp_fa/${GROUP}_wrf.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "523.xalancbmk_r" --group "$GROUP" > ./tmp_fa/${GROUP}_xalan.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "525.x264_r" --group "$GROUP" > ./tmp_fa/${GROUP}_x264.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "526.blender_r" --group "$GROUP" > ./tmp_fa/${GROUP}_blender.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "527.cam4_r" --group "$GROUP" > ./tmp_fa/${GROUP}_cam4.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "531.deepsjeng_r" --group "$GROUP" > ./tmp_fa/${GROUP}_deepsjeng.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "538.imagick_r" --group "$GROUP" > ./tmp_fa/${GROUP}_imagick.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "541.leela_r" --group "$GROUP" > ./tmp_fa/${GROUP}_leela.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "544.nab_r" --group "$GROUP" > ./tmp_fa/${GROUP}_nab.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "548.exchange2_r" --group "$GROUP" > ./tmp_fa/${GROUP}_exchange2.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "549.fotonik3d_r" --group "$GROUP" > ./tmp_fa/${GROUP}_fotonik3d.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "554.roms_r" --group "$GROUP" > ./tmp_fa/${GROUP}_roms.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.feature_analysis --test-file "557.xz_r" --group "$GROUP" > ./tmp_fa/${GROUP}_xz.out &

wait

cat ./tmp_fa/*.out > feature_analysis_${GROUP}.out
rm ./tmp_fa/*.out
