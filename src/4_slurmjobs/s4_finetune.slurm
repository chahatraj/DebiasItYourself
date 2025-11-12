#!/bin/bash

#SBATCH --job-name=ft_ci
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=90GB
#SBATCH --time=0-01:59:58
#SBATCH --array=4
#SBATCH --output=/scratch/craj/logs/cog/out/ft_all.%A_%a.out.txt
#SBATCH --error=/scratch/craj/logs/cog/err/ft_all.%A_%a.err.txt

#nvidia-smi
cd /scratch/craj/diy/src/3_experiments

# Read arguments from the config file
readarray -t ARGS < /scratch/craj/diy/src/4_slurmjobs/s4.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

#echo $ARGUMENTS
echo "Task $SLURM_ARRAY_TASK_ID → $ARGUMENTS"

python 7_finetunellama.py $ARGUMENTS
