#!/bin/bash
#SBATCH -A nph123
#SBATCH -N 2
#SBATCH -J evc_cc
#SBATCH -t 1:00:00
#SBATCH -o output%J
#SBATCH -e error%J
#SBATCH -p gpu

export OMP_NUM_THREADS=4

date

# for intel
module purge
module load DefApps
module load hdf5/1.10.6
module load intel
module load gsl

srun -n4 -N1 ./prog_cc_jscheme.x split_input_Ct1S0nn.ini > split_input_Ct1S0nn.out &
srun -n4 -N1 ./prog_cc_jscheme.x split_input_C3P0.ini > split_input_C3P0.out &

wait