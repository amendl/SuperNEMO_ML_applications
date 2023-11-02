#!/bin/bash

#SBATCH --partition=htc                  # Partition choice (most generally we work with htc, but for quick debugging you can use flash
#SBATCH --output=stdout3.log
#SBATCH -e stdout3.log
										 #					 #SBATCH --partition=htc. This avoids waiting times, but is limited to 1hr)
#SBATCH --licenses=sps                   # When working on sps, must declare license!!

source /pbs/software/centos-7-x86_64/root/"6.22.06-fix01"/bin/ccenv.sh
source /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/activate

export P1=0
export P2=0
export P3=8
export P4=0
export P5=0

# Copy files
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_0.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_1.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_2.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_3.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_4.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_5.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_6.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_7.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_8.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/2/my_generator_t2_9.root .

cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_0.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_1.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_2.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_3.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_4.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_5.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_6.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_7.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_8.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/1/my_generator_t1_9.root .

cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_0.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_1.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_2.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_3.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_4.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_5.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_6.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_7.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_8.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/3/my_generator_t3_9.root .

cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_0.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_1.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_2.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_3.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_4.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_5.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_6.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_7.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_8.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_with_hint/4/my_generator_t4_9.root .

python ../../scripts/train_binary_loss.py --OneDeviceStrategy "/gpu:0"

# python ../../scripts/test.py binary

