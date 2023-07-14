

# Goals
 * Calculate number of tracks in event by CNN (1, 2, 3 or 4 tracks).
 * Use generative adversarial networks with convolutional autoencoders for track clustering.
 * Predict associated calorimeter hit(s) using single-label classification on clustered events or multi-label classification on not clustered events.
# Software
This should work without problems on CCLyon in2p3 computation cluster.
## Required software
Almost everything runs ot top of `python3`. On CCLyon use `python` sourced with `root` via
1. `ccenv root 6.22.06` - loads `python 3.8.6` (**does not work now**)
2. since July 12 2023 `module add Analysis/root/6.22.06-fix01` - loads `python 3.9.1` (**currently, this is the way to go**)

This software should be installed in python or Anaconda environment (python environment is prefered since it can access  both sourced root package and all gpu related software directly, however it is still possible to make it work with Anaconda)
 * `Root` - Root is not needed to be explicitly installed in python or Anaconda environment, any sourced Root on CCLyon should work - minimum tested verion 6.22.06 (since July 12 2021 6.22.06-fix01)
 * `cudatoolkit`, `cudnn` - Should be already installed on CCLyon 
 * `tensorflow`
 * `keras` - Should be part of `tensorflow`
 * `numpy`
 * `maplotlib`, `seaborn`
 * `scikit-learn`
 * `pydot`, `graphviz`
 * `argparse`
 
Optional:
 * `tensorboard` - Should be part of `tensorflow`
 * `tensorboard_plugin_profile` - Should be part of `tensorflow`
 * `nvidia-pyindex`, `nvidia-tensorrt` - For TensorRT support
 * `nvidia-smi` -  For checking usage and available memory on NVIDIA V100 GPU (on CCLyon)

## Working with real data
We test models on real data and compared them with [TKEvent](https://github.com/TomasKrizak/TKEvent). Unfortunately, it is not possible to open `root` files produced by [TKEvent](https://github.com/TomasKrizak/TKEvent) library since this library might be built with different version of python and libstdc++. Fortunately, workaround exists. We need to download and build two versions of [TKEvent](https://github.com/TomasKrizak/TKEvent). First version will be built in the manner described in [TKEvent](https://github.com/TomasKrizak/TKEvent) README.md. The second library shoudl be build (we ignore the `red_to_tk` target) with following steps:

1.`module add ROOT` where `ROOT` is version of `root` library used by `tensorflow`
2. `TKEvent/TKEvent/install.sh` to build library 

Now, we can use `red_to_tk` from the first library to obtain root file with `TKEvent` objects and open this root file with the second version of `TKEvent` library.
## Running scripts (on CCLyon in2p3 cluster)
Example is at `example_exec.sh`. Run it with `sbatch --mem=... -n 1 -t ... gres=gpu:v100:N example_exec.sh` if you have access to GPU, where `N` is number of GPUs you want to use. Otherwise, leave out `gres` option.

Scripts can use two strategies. To use only one GPU use option `--OneDeviceStrategy "/gpu:0"`. If you want to use more GPUs, use for example `--MirroredStrategy "/gpu:0" "/gpu:1" "/gpu:2"`.
## Workflow overview
1. source `python` and `root` -  `module add Analysis/root/6.22.06-fix01`
2. create python virtual environment (if not done yet) 
3. install [packages](#required-software) (if not done yet)
4. load python virtual environment
# Description of files
 * `lib.py` -  small library with some functions that are reused across this project 
 * `number_of_tracks_classification.py`
 * `combined.py` - Script for constructing and training model consisting from `top`, `side` and `front` models.
 * `number_of_tracks_classification3D.py` - TODO: Classification using Conv3D
 * `plot_confusion.py` - Script helping analyze badly classified events
 * `clustering_one.py`- example of custom trainig loop for GAN autoencoders used for clustering
 * `multilabel_analysis.py` - script for analysing various aspect of multilabel classifier used for associated calorimeter hit detection
## `capsule`
 * `routing_by_agreement.py` - tensorflow.while_loop implementation of routing algorithm
 * `capsule_lib.py` - basic parts of CapsNET architecture
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
 * `top` - Number of tracks classifier viewing detector from top (`SN-IEGenerator`)
 * `side` - Number of tracks classifier viewing detector from side (`SN-IEGenerator`)
 * `front` - Number of tracks classifier viewing detector from front (`SN-IEGenerator`)
 * `combined` - Top, side and front view combined usign transfer learning (`SN-IEGenerator`)
 * `top_my_generator` - Number of tracks classifier viewing detector from top (`my_generator`)
 * `side_my_generator ` - Number of tracks classifier viewing detector from side (`my_generator`)
 * `front_my_generator ` - Number of tracks classifier viewing detector from front (`my_generator`)
 * `combined_my_generator ` - Top, side and front view combined usign transfer learning (`my_generator`)
# Results (trained and tested on SN-IEGenerator, my_generator)
 * [Confusion matrix for combined model (SN-IEGenerator)](./ImagesAndDocuments/combined.pdf)
 * [Confusion matrix for top model (my_generator)](./ImagesAndDocuments/top_model_my_generator_confusion_matrix.pdf)
 * [Confusion matrix for side model (my_generator)](./ImagesAndDocuments/side_model_my_generator_confusion_matrix.pdf)
 * [Confusion matrix for front model (my_generator)](./ImagesAndDocuments/front_model_my_generator_confusion_matrix.pdf)
 * [Confusion matrix for combined model (my_generator)](./ImagesAndDocuments/combined_model_my_generator_confusion_matrix.pdf)
# Results (on real data)

# Issues
 * sbatch and tensorflow sometimes fail to initialize libraries (mainly to source python from virtual environment or root) - start the script again
 * tensorflow sometimes runs out of memory - don't use checkpoints for tensorboard
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
 * [MATRIX CAPSULES WITH EM ROUTING](https://openreview.net/pdf?id=HJWLfGWRb)
## Tensorflow
 * [Dense layer in keras (how to build custom keras model)](https://github.com/keras-team/keras/blob/v2.12.0/keras/layers/core/dense.py)
 * [Tricks for custom layers](https://oliver-k-ernst.medium.com/a-cheat-sheet-for-custom-tensorflow-layers-and-models-aa465df2bc8b)
## Classification
 * [Facebook, Instagram: Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/pdf/1805.00932.pdf) - [Discussion #1](https://stats.stackexchange.com/questions/433867/why-is-softmax-considered-counter-intuitive-for-multi-label-classification), [Discussion #2](https://stackoverflow.com/questions/66990074/using-softmax-for-multilabel-classification-as-per-facebook-paper)
   * how to use softmax for multi-label classification
 * [Bags of Tricks for Multi-Label Classification](https://andy-wang.medium.com/bags-of-tricks-for-multi-label-classification-dc54b87f79ec)
 * [Discussion about Tensorflow losses](https://stats.stackexchange.com/questions/207794/what-loss-function-for-multi-class-multi-label-classification-tasks-in-neural-n)
 * [A no-regret generalization of hierarchical softmax to extreme multi-label classification](https://proceedings.neurips.cc/paper_files/paper/2018/file/8b8388180314a337c9aa3c5aa8e2f37a-Paper.pdf)
 * [Metrics for Multi-Label Classification](https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics)
## Classifier results visualization
 * good overview: [Visual Comparison of Multi-label Classification Results](https://diglib.eg.org/bitstream/handle/10.2312/vmv20211367/017-026.pdf?sequence=1&isAllowed=y)
 * Single-label multi-class classification [ComDia+](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8781593)
 * Multi-lable classification [UnTangleMap](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7091015) - **can show only one result at a time**
### Set-based visualizations
 Label is set, data samples, to which label was assignet are elements of the set
 * [UpSet]()
 * [AggreSet]()
## Random stuff
 * [Transformer position as dynamical model](https://assets.amazon.science/f0/32/ff7d9669492bbe2dedb8ee3cb3e5/learning-to-encode-position-for-transformer-with-continuous-dunamical-model.pdf)
 * [Neural Ordinary Differential Equations](https://papers.nips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html)
 * [Learning Jacobian Trace of Ordinary Differential Equation](https://arxiv.org/pdf/1810.01367)

---
If you have any questions feel free to contact me at [adam.mendl@cvut.cz](mailto:adam.mendl@cvut.cz) or [amendl@hotmail.com](mailto:amendl@hotmail.com).
