#!/bin/bash

#SBATCH --job-name=gen_deb
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=90GB
#SBATCH --time=0-03:59:58
#SBATCH --array=1-5
#SBATCH --output=/scratch/craj/logs/diy/out/bbq_gen_n.%A_%a.out.txt
#SBATCH --error=/scratch/craj/logs/diy/err/bbq_gen_n.%A_%a.err.txt

#nvidia-smi
cd /scratch/craj/diy/src/2_generations

# Read arguments from the config file
readarray -t ARGS < /scratch/craj/diy/src/4_slurmjobs/s3.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

#echo $ARGUMENTS
echo "Task $SLURM_ARRAY_TASK_ID → $ARGUMENTS"

python 6_gen_debiased_instances.py $ARGUMENTS
