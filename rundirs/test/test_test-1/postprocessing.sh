#!/bin/bash
#PBS -N full20-pprocess
#PBS -P v45
#PBS -q normalbw
#PBS -l mem=112gb
#PBS -l walltime=6:00:00
#PBS -l ncpus=28
#PBS -l storage=gdata/v45+scratch/v45+scratch/x77+gdata/v45+gdata/nm03+gdata/hh5+scratch/nm03
cd $PBS_O_WORKDIR
CWD=$(pwd)
PYTHONNOUSERSITE=1
module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
module list
python3 ../../../postprocess.py -e "$CWD"
