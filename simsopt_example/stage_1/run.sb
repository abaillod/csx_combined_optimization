#!/bin/sh

#SBATCH --account=apam
#SBATCH -N 1                 # Recommended by Grace
#SBATCH --ntasks-per-node=32 # Maximum
#SBATCH --job-name=simsopt_examplen
#SBATCH --time=0-12:00       # Might be too small. Can restart from an iteration if necessary
#SBATCH --mem-per-cpu=5G     # Maximum - Ginsburg has 187G available per node

source ~/.bashrc
conda activate simsopt
module load gcc/10.2.0
module load openmpi/gcc/64/4.1.5a1
module load netcdf/gcc/64/gcc/64/4.7.4
module load lapack/gcc/64/3.9.0
module load hdf5p/1.10.7
module load netcdf-fortran/4.5.3 
module load intel-parallel-studio/2020

srun --mpi=pmix_v3 python single_stage_optimization.py
