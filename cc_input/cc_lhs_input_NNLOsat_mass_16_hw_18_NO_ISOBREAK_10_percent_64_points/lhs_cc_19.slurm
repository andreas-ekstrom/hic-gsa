#!/bin/bash
#SBATCH -A nph123
#SBATCH -N 2
#SBATCH -J evc_cc
#SBATCH -t 1:00:00
#SBATCH -o output%J
#SBATCH -e error%J
#SBATCH -p gpu

export OMP_NUM_THREADS=4
work_dir=/lustre/orion/scratch/hagen/nph123/vint_runs
cd $work_dir

date

# for intel
module purge
module load DefApps
module load hdf5/1.10.6
module load intel
module load gsl

srun -n4 -N1 ./prog_cc_jscheme.x cc_lhs_input_NNLOsat_mass_16_hw_18_NO_ISOBREAK_10percent_38_64_pnts.ini > cc_lhs_input_NNLOsat_mass_16_hw_18_NO_ISOBREAK_10percent_38_64_pnts.out &
srun -n4 -N1 ./prog_cc_jscheme.x cc_lhs_input_NNLOsat_mass_16_hw_18_NO_ISOBREAK_10percent_39_64_pnts.ini > cc_lhs_input_NNLOsat_mass_16_hw_18_NO_ISOBREAK_10percent_39_64_pnts.out &

wait