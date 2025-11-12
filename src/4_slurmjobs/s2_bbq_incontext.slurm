#!/bin/bash

#SBATCH --job-name=bbq_inc
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --time=0-02:59:58
#SBATCH --array=24,26,33,35,42,44,51,53,57,60,62,69,71,78,80,87,89
#SBATCH --output=/scratch/craj/logs/diy/out/bbq_incontext_n.%A_%a.out.txt
#SBATCH --error=/scratch/craj/logs/diy/err/bbq_incontext_n.%A_%a.err.txt

#nvidia-smi
cd /scratch/craj/diy/src/3_experiments

# Read arguments from the config file
readarray -t ARGS < /scratch/craj/diy/src/4_slurmjobs/s2.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

#echo $ARGUMENTS
echo "Task $SLURM_ARRAY_TASK_ID → $ARGUMENTS"

python 3_bbq_incontext.py $ARGUMENTS
