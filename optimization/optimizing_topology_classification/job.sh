#!/bin/bash

#SBATCH --partition=htc                  # Partition choice (most generally we work with htc, but for quick debugging you can use flash
#SBATCH --output=stdout.log
#SBATCH -e stdout.log
										 #					 #SBATCH --partition=htc. This avoids waiting times, but is limited to 1hr)
#SBATCH --licenses=sps                   # When working on sps, must declare license!!

source /pbs/software/centos-7-x86_64/root/"6.22.06-fix01"/bin/ccenv.sh
source /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/activate


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

cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_0_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_1_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_2_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_3_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_4_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_5_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_6_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_7_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_8_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/2/my_generator_t2_9_signal_like.root .

cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_0_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_1_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_2_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_3_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_4_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_5_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_6_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_7_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_8_signal_like.root .
cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/1/my_generator_t1_9_signal_like.root .

# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_0_signal_like.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_1_signal_like.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_2_signal_like.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_3_signal_like.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_4_signal_like.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_5_signal_like.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_6_signal_like.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_7_signal_like.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_8_signal_like.root .
# cp /sps/nemo/scratch/amendl/AI/datasets/my_generator_signal_like/3/my_generator_t3_9_signal_like.root .



python ../../scripts/train_binary_loss.py
