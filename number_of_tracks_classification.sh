#!/bin/bash

#SBATCH --partition=htc                  # Partition choice (most generally we work with htc, but for quick debugging you can use flash
#SBATCH --output=stdout.log
#SBATCH -e stdout.log
										 #					 #SBATCH --partition=htc. This avoids waiting times, but is limited to 1hr)
#SBATCH --licenses=sps                   # When working on sps, must declare license!!



# export PATH=/pbs/software/centos-7-x86_64/anaconda/3.6/condabin:$PATH
# conda run -n myenv python3 -c "import sys; print (sys.version); import tensorflow as tf; print(tf.config.list_physical_devices());"
# lspci | grep -i nvidia
# echo "Hello world"


source /pbs/software/centos-7-x86_64/root/"6.22.06"/bin/ccenv.sh
source /sps/nemo/scratch/amendl/AI/my_new_env/bin/activate

python ../number_of_tracks_classification.sh