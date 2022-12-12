#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 4
#SBATCH --mem=200g
#SBATCH -w arctrddgx004
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:V100:1
#SBATCH -t 02-00
#SBATCH -J wandbtemp
#SBATCH -e error%A.err
#SBATCH -o out%A.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bthapaliya1@student.gsu.edu
#SBATCH --oversubscribe

sleep 10s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo $HOSTNAME >&2 

source /data/users2/bthapaliya/anaconda-main/anaconda3/bin/activate 

wandb agent exposingbrain/hcp_custom/v0obg0il
sleep 30s