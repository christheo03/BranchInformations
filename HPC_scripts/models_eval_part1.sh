#!/bin/bash
#SBATCH --job-name=Models-Eval-1
#SBATCH --partition=GPU,INFERENCE
#SBATCH --mem=20G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --error=/home/ctheod03/repo/BranchInformations/logs/%x_%j.err

source /home/ctheod03/.local/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate env


export PYTHONPATH=/home/ctheod03/repo/BranchInformations:/home/ctheod03/repo/BranchInformations/mlp_predictor

cd /home/ctheod03/repo/BranchInformations/mlp_predictor

mkdir -p ./tmp1

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "505.mcf_r" --min-executed 1 > ./tmp1/embmcf.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "500.perlbench_r" --min-executed 1 > ./tmp1/embperl.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "502.gcc_r" --min-executed 1 > ./tmp1/embgcc.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "520.omnetpp_r" --min-executed 1 > ./tmp1/embomnet.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "523.xalancbmk_r" --min-executed 1 > ./tmp1/embxalan.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "507.cactuBSSN_r" --min-executed 1 > ./tmp1/embcactu.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "508.namd_r" --min-executed 1 > ./tmp1/embnamd.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "510.parest_r" --min-executed 1 > ./tmp1/embparest.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "511.povray_r" --min-executed 1 > ./tmp1/embpovray.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "519.lbm_r" --min-executed 1 > ./tmp1/emblbm.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "521.wrf_r" --min-executed 1 > ./tmp1/embwrf.out &

 python -u -m mlp_predictor.Taken_NotTaken.EMB_CT --test-file "525.x264_r" --min-executed 1 > ./tmp1/embx264.out &

wait

# Combine all the temporary files into the final file
cat ./tmp1/*.out > wmr_n1_exe3.out

# Delete the temporary files so your folder stays clean
rm ./tmp1/*.out
