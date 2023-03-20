# SuperNEMO_ML_applications
## Goals
## Required software
This software should be installed in python or Anaconda environment
 * `Root` - Root is not needed to be explicitly install in python environment, any sourced Root on CCLyon should work (minimum tested verion 6.22.06)
 * `cudatoolkit`, `cudnn` - Should be installed on CCLyon 
 * `TensorFlow`
 * `Keras` - Should be part of tensorflow
 * `numpy`
 * `maplotlib`
Optional:
 * `nvidia-pyindex`, `nvidia-tensorrt` for TensorRT support
## Running scripts
 Example is at `example_exec.sh`. Run it with `sbatch --mem=... -n 1 -t ... gres=gpu:v100:1 example_exec.sh` if you have access to GPU. Otherwise, leave out `gres` option.

# Description of files
 * `number_of_tracks_classification.py`
 * `combined.py` - Script for constructing and training model consisting from `top`, `side` and `front` models.
 * `number_of_tracks_classification3D.py` - TODO: Classification using Conv3D
## `Generator`
This folder contains files from [SN-IEGenerator](https://github.com/SuperNEMO-DBD/SN-IEgenerator) (version from Mar 7, 2018) that were modified for out project. 
 * `toyhaystack.py` - Clustering of hits into tracks added.
## `datasets`
This folder contains essential scripts for loading and preprocessing data
 * `number_of_tracks_classification.py` - If you want to change folder with training and testing data, see line 20.
## Models
Trained models in TensorFlow format.
 * `top`
 * `side`
 * `front`
 * `conbined`
---
If you have any questions feel free to contact me at [adam.mendl@cvut.cz](mailto:adam.mendl@cvut.cz).
