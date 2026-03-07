#!/bin/bash
#SBATCH --job-name=HMM_nose
#SBATCH --partition=gpu
#SBATCH --account=smearlab
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-80gb,no-mig
#SBATCH --cpus-per-task=8
#SBATCH --output=./out_hmm%j
#SBATCH --error=./errors_hmm%j


source ~/miniconda3/etc/profile.d/conda.sh
conda activate HMM
export PATH="/home/trevorro/miniconda3/envs/HMM/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH"
which ptxas || { echo "FATAL: ptxas not found"; exit 1; }

python /projects/smearlab/trevorro/scripts/project/fithmm.py \
    --data-dir  /projects/smearlab/trevorro/data/events \
    --output-dir /projects/smearlab/trevorro/results/hmm_${SLURM_JOB_ID} \
    --n-states   5 6 7 8 10 12 \
    --n-restarts 4 \
    --n-em-iters 500

