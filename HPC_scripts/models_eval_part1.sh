#!/bin/bash
#SBATCH --job-name=Models-Eval-1
#SBATCH --partition=GPU
#SBATCH --mem=58G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gpu03
#SBATCH --error=/home/ctheod03/repo/BranchInformations/logs/%x_%j.err

source /home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate env


export PYTHONPATH=/home/ctheod03/repo/BranchInformations:/home/ctheod03/repo/BranchInformations/mlp_predictor

cd /home/ctheod03/repo/BranchInformations/mlp_predictor

mkdir -p ./tmp1

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "505.mcf_r" --min-executed 1000 > ./tmp1/espmcf.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "505.mcf_r" --min-executed 1000 > ./tmp1/ssmcf.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "505.mcf_r" --min-executed 1000 > ./tmp1/embmcf.out &

CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "500.perlbench_r" --min-executed 1000 > ./tmp1/espperl.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "500.perlbench_r" --min-executed 1000 > ./tmp1/ssperl.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "500.perlbench_r" --min-executed 1000 > ./tmp1/embperl.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "502.gcc_r" --min-executed 1000 > ./tmp1/espgcc.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "502.gcc_r" --min-executed 1000 > ./tmp1/ssgcc.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "502.gcc_r" --min-executed 1000 > ./tmp1/embgcc.out &

CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "520.omnetpp_r" --min-executed 1000 > ./tmp1/espomnet.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "520.omnetpp_r" --min-executed 1000 > ./tmp1/ssomnet.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "520.omnetpp_r" --min-executed 1000 > ./tmp1/embomnet.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "523.xalancbmk_r" --min-executed 1000 > ./tmp1/espxalan.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "523.xalancbmk_r" --min-executed 1000 > ./tmp1/ssxalan.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "523.xalancbmk_r" --min-executed 1000 > ./tmp1/embxalan.out &

CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "507.cactuBSSN_r" --min-executed 1000 > ./tmp1/espcactu.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "507.cactuBSSN_r" --min-executed 1000 > ./tmp1/sscactu.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "507.cactuBSSN_r" --min-executed 1000 > ./tmp1/embcactu.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "508.namd_r" --min-executed 1000 > ./tmp1/espnamd.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "508.namd_r" --min-executed 1000 > ./tmp1/ssnamd.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "508.namd_r" --min-executed 1000 > ./tmp1/embnamd.out &

CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "510.parest_r" --min-executed 1000 > ./tmp1/espparest.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "510.parest_r" --min-executed 1000 > ./tmp1/ssparest.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "510.parest_r" --min-executed 1000 > ./tmp1/embparest.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "511.povray_r" --min-executed 1000 > ./tmp1/esppovray.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "511.povray_r" --min-executed 1000 > ./tmp1/sspovray.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "511.povray_r" --min-executed 1000 > ./tmp1/embpovray.out &

CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "519.lbm_r" --min-executed 1000 > ./tmp1/esplbm.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "519.lbm_r" --min-executed 1000 > ./tmp1/sslbm.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "519.lbm_r" --min-executed 1000 > ./tmp1/emblbm.out &

CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "521.wrf_r" --min-executed 1000 > ./tmp1/espwrf.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "521.wrf_r" --min-executed 1000 > ./tmp1/sswrf.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "521.wrf_r" --min-executed 1000 > ./tmp1/embwrf.out &

CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.ESP --test-file "525.x264_r" --min-executed 1000 > ./tmp1/espx264.out &
CUDA_VISIBLE_DEVICES=0 python -u -m mlp_predictor.Taken_NotTaken.SS_CT --test-file "525.x264_r" --min-executed 1000 > ./tmp1/ssx264.out &
CUDA_VISIBLE_DEVICES=1 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "525.x264_r" --min-executed 1000 > ./tmp1/embx264.out &

wait

# Combine all the temporary files into the final file
cat ./tmp1/*.out > smr_n1_exe1000.out

# Delete the temporary files so your folder stays clean
rm ./tmp1/*.out
