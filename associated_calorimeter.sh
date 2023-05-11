#!/bin/bash

#SBATCH --partition=htc                  # Partition choice (most generally we work with htc, but for quick debugging you can use flash
#SBATCH --output=stdout.log
#SBATCH -e stdout.log
										 #					 #SBATCH --partition=htc. This avoids waiting times, but is limited to 1hr)
#SBATCH --licenses=sps                   # When working on sps, must declare license!!

source /sps/nemo/scratch/amendl/AI/virtual211/bin/activate

python ../associated_calorimeter.py --OneDeviceStrategy "/gpu:0"
