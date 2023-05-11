# Machine learning for SuperNEMO tracking and recontruction
## Goals
 * Calculate number of tracks in event by CNN.
 * Use generative adversarial networks with convolutional autoencoders for track clustering.
 * Predict associated calorimeter hit(s) using single-label classification on clustered events or multi-label classification on not clustered events.
## Required software
This software should be installed in python or Anaconda environment
 * `Root` - Root is not needed to be explicitly installed in python or Anaconda environment, any sourced Root on CCLyon should work (minimum tested verion 6.22.06)
 * `cudatoolkit`, `cudnn` - Should be already installed on CCLyon 
 * `tensorflow`
 * `keras` - Should be part of `tensorflow`
 * `numpy`
 * `maplotlib`, `seaborn`
Optional:
 * `tensorboard` - Should be part of `tensorflow`
 * `tensorboard_plugin_profile` - Should be part of `tensorflow`
 * `nvidia-pyindex`, `nvidia-tensorrt` - For TensorRT support
 * `nvidia-smi` -  For checking usage and available memory on NVIDIA V100 GPU (on CCLyon)
## Running scripts (on CCLyon in2p3 cluster)
Example is at `example_exec.sh`. Run it with `sbatch --mem=... -n 1 -t ... gres=gpu:v100:N example_exec.sh` if you have access to GPU, where `N` is number of GPUs you want to use. Otherwise, leave out `gres` option.

Scripts can use two strategies. To use only one GPU use option `--OneDeviceStrategy "/gpu:0"`. If you want to use more GPUs, use for example `--MirroredStrategy "/gpu:0" "/gpu:1" "/gpu:2"`.

# Description of files
 * `number_of_tracks_classification.py`
 * `combined.py` - Script for constructing and training model consisting from `top`, `side` and `front` models.
 * `number_of_tracks_classification3D.py` - TODO: Classification using Conv3D
 * `plot_confusion.py` - Script helping analyze badly classified events
 * `clustering_one.py`
## `capsule`
 * `routing_by_agreement.py` - tensorflow.while_loop implementation of routing algorithm
 * `capsule_lib.py` - 
## `architectures`
 * `top`
 * `side`
 * `front`
 * `generator` - Generator used in GAN architecture
 * `discriminator` - Discriminator used in GAN architecture
## `Generator`
This folder contains files from [SN-IEGenerator](https://github.com/SuperNEMO-DBD/SN-IEgenerator) (version from Mar 7, 2018) that were modified for out project. 
 * `toyhaystack.py` - Clustering of hits into tracks added.
## `my_generator`
 * `my_generator.cxx` - Alternative for toyhaystack generator in form of root script.
 * `generate.sh` - sample bash script for using `my_generator.cxx` 
## `datasets`
This folder contains essential scripts for loading and preprocessing data
 * `number_of_tracks.py` - If you want to change folder with training and testing data, see line 20.
 * `number_of_tracks3D.py` - TODO
 * `clustering.py`
 * `associated_calohit_multilabel.py`
 * `associated_calohit_singlelabel.py`
## `models`
Trained models in TensorFlow format.
 * `top` - Number of tracks classifier viewing detector from top
 * `side` - Number of tracks classifier viewing detector from side
 * `front` - Number of tracks classifier viewing detector from front
 * `combined` - Top, side and front view combined usign transfer learning
# Results
* [Confusion matrix for combined model](./ImagesAndDocuments/combined.pdf)

# Resources
## Convolutional neural networks and autoencoders
 * [Multi-label image classification](https://towardsdatascience.com/multi-label-image-classification-with-neural-network-keras-ddc1ab1afede)
 * [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)
 * [Building a simple Generative Adversarial Network (GAN) using TensorFlow](https://blog.paperspace.com/implementing-gans-in-tensorflow/)
 * [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf)
## CapsNET
 * [Series of articles on Capsule architecture](https://pechyonkin.me/capsules-1/)
 * [Preprint on routing algorithm (Capsule architecture)](https://arxiv.org/abs/1710.09829)
 * [Tensorflow implementation 1](https://www.kaggle.com/code/giovanimachado/capsnet-tensorflow-implementation)
 * [Tensorflow implementation 2](https://towardsdatascience.com/implementing-capsule-network-in-tensorflow-11e4cca5ecae)


---
If you have any questions feel free to contact me at [adam.mendl@cvut.cz](mailto:adam.mendl@cvut.cz) or [amendl@hotmail.com](mailto:amendl@hotmail.com).
