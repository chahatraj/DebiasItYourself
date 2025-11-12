#!/bin/bash

#SBATCH --job-name=bbq_inf
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --time=0-00:59:58
#SBATCH --array=1-9
#SBATCH --output=/scratch/craj/logs/diy/out/bbq_inference_n.%A_%a.out.txt
#SBATCH --error=/scratch/craj/logs/diy/err/bbq_inference_n.%A_%a.err.txt

#nvidia-smi
cd /scratch/craj/diy/src/3_experiments

# Read arguments from the config file
readarray -t ARGS < /scratch/craj/diy/src/4_slurmjobs/s1.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

#echo $ARGUMENTS
echo "Task $SLURM_ARRAY_TASK_ID → $ARGUMENTS"

python ft_inference.py $ARGUMENTS
