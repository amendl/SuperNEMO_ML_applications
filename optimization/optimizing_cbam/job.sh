#!/bin/bash

#SBATCH --partition=htc                  # Partition choice (most generally we work with htc, but for quick debugging you can use flash
#SBATCH --output=stdout3.log
#SBATCH -e stdout3.log
										 #					 #SBATCH --partition=htc. This avoids waiting times, but is limited to 1hr)
#SBATCH --licenses=sps                   # When working on sps, must declare license!!

source /pbs/software/centos-7-x86_64/root/"6.22.06-fix01"/bin/ccenv.sh
source /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/activate


# Copy files
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_0.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_1.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_2.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_3.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_4.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_5.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_6.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_7.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_8.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_9.root .

# python ../../scripts/train_binary_loss.py --OneDeviceStrategy "/cpu:0"

python ../../scripts/test.py binary

