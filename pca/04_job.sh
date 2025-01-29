#!/bin/bash
#SBATCH -N1 --ntasks-per-node=1   --exclusive
#SBATCH --output=log/out.%j
#SBATCH --error=log/err.%j
#SBATCH --time=10:00:00
#SBATCH --partition=ccq
#SBATCH --mem=400GB

#======START=====


# Arguments for start and end values
#module purge
module load python

source /mnt/home/jzang/jax_env/bin/activate


python3 -u pca_fb_cal.py  > log/pca00.out
