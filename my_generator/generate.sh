#! /bin/bash

#SBATCH -J my_generator
#SBATCH --licenses=sps
#SBATCH -e stdout.log
#SBATCH -o stdout.log

ccenv root 6.22.06

EVENTS_IN_FILE=20000
STARTING_FILE_NUMBER=0
ENDING_FILE_NUMBER=10
# lines: 1
LINES=1
for i in {0..9}
do
  root -b -q -l "my_generator.cxx($LINES,$EVENTS_IN_FILE,\"$LINES/my_generator\",$i)"
done

# lines: 2
LINES=2
for i in {0..9}
do
  root -b -q -l "my_generator.cxx($LINES,$EVENTS_IN_FILE,\"$LINES/my_generator\",$i)"
done

# lines: 3
LINES=3
for i in {0..9}
do
  root -b -q -l "my_generator.cxx($LINES,$EVENTS_IN_FILE,\"$LINES/my_generator\",$i)"
done

# lines: 4
LINES=4
for i in {0..9}
do
  root -b -q -l "my_generator.cxx($LINES,$EVENTS_IN_FILE,\"$LINES/my_generator\",$i)"
done