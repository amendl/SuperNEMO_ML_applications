#!/bin/bash

#SBATCH --partition=htc
#SBATCH -o stdout.log
#SBATCH -e stdout.log					 
#SBATCH --licenses=sps

module purge
source /pbs/software/centos-7-x86_64/root/"6.22.06-fix01"/bin/ccenv.sh
source /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/activate

python ../clustering_matteo.py --OneDeviceStrategy "/gpu:0"