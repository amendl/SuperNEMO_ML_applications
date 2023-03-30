# Machine learning for SuperNEMO tracking and recontruction
## Goals
 * Calculate number of tracks in event by CNN.
 * Use generative adversarial networks with convolutional autoencoders for track clustering.
 * Predict associated calorimeter hit(s) using single-label classification on clustered events or multi-label classification on not clustered events.
## Required software
This software should be installed in python or Anaconda environment
 * `Root` - Root is not needed to be explicitly installed in python environment, any sourced Root on CCLyon should work (minimum tested verion 6.22.06)
 * `cudatoolkit`, `cudnn` - Should be already installed on CCLyon 
 * `TensorFlow`
 * `Keras` - Should be part of Tensorflow
 * `numpy`
 * `maplotlib`, `seaborn`

Optional:
 * `nvidia-pyindex`, `nvidia-tensorrt` for TensorRT support
 * `nvidia-smi` for checking usage and available memory on NVIDIA V100 GPU
## Running scripts
Example is at `example_exec.sh`. Run it with `sbatch --mem=... -n 1 -t ... gres=gpu:v100:1 example_exec.sh` if you have access to GPU. Otherwise, leave out `gres` option.

# Description of files
 * `number_of_tracks_classification.py`
 * `combined.py` - Script for constructing and training model consisting from `top`, `side` and `front` models.
 * `number_of_tracks_classification3D.py` - TODO: Classification using Conv3D
 * `plot_confusion.py` - Script helping analyze badly classified events
 * `clustering_one.py`
## `architectures`
 * `top`
 * `side`
 * `front`
 * `generator` - Generator used in GAN architecture
 * `discriminator` - Discriminator used in GAN architecture
## `Generator`
This folder contains files from [SN-IEGenerator](https://github.com/SuperNEMO-DBD/SN-IEgenerator) (version from Mar 7, 2018) that were modified for out project. 
 * `toyhaystack.py` - Clustering of hits into tracks added.
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

# Resources about Convolutional neural networks and autoencoders
 * [Series of articles on Capsule architecture](https://pechyonkin.me/capsules-1/)
 * [Preprint on routing algorithm (Capsule architecture)](https://arxiv.org/abs/1710.09829)
 * [Multi-label image classification](https://towardsdatascience.com/multi-label-image-classification-with-neural-network-keras-ddc1ab1afede)
 * [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)
 * [Building a simple Generative Adversarial Network (GAN) using TensorFlow](https://blog.paperspace.com/implementing-gans-in-tensorflow/)
 * [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf)



---
If you have any questions feel free to contact me at [adam.mendl@cvut.cz](mailto:adam.mendl@cvut.cz).
