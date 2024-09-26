#!/bin/bash
#PBS -N getefs
#PBS -P ol01
#PBS -q normal
#PBS -l mem=112gb
#PBS -l walltime=1:00:00
#PBS -l ncpus=28
#PBS -l storage=gdata/v45+scratch/v45+scratch/x77+gdata/v45+gdata/nm03+gdata/hh5+scratch/nm03

module use /g/data/hh5/public/modules
module load conda/analysis3-24.04
module list


python3 /home/149/ab8992/topographic-NIWs -e "height" 
