
You can look into `presentation` folder for less technicaly detailed overview.

Note that if we are writing about **clustering tracker hits**, we are writing about image segmentation int the terms of ML community.

If you have any questions feel free to contact me at [adam.mendl@cvut.cz](mailto:adam.mendl@cvut.cz) or [amendl@hotmail.com](mailto:amendl@hotmail.com).

---
# Goals
 * Calculate number of tracks in event by CNN (1, 2, 3 or 4 tracks).
 * Use generative adversarial networks with convolutional autoencoders for track clustering.
 * Predict associated calorimeter hit(s) using single-label classification on clustered events or multi-label classification on not clustered events.
# Progress
 * Discussing with Tomas how to incorporate ML into his fitting algorithm.
 * Tested track counting model on real data. Summary of all efforts is that the model adapts **pretty well** on measured data. 
 * First parts of CapsNET layer, hoping to finish this soon.
 * **Currently working on autoencoders** - I have done some simple tests with Matteos architecture. Now trying to use variational autoencoders.
## Calculating number of tracks
 * Counting tracks done on my_generator (works really well). It removes the problem that [SN-IEGenerator](https://github.com/SuperNEMO-DBD/SN-IEgenerator) sometimes generates event with less tracks.
 * Tested on real data, two main problems:
   1. isolated tracker hits are detected as one track,
   2. if the z position of tracker hit cannot be computed, then it is set to zero. Side and Front views of this model detect those as aditional track
 * Primitive solutions applied (use only top view and filter isolated tracker hits) and then it works well. See [Results](#results) section.
 * Reformulation of this task to finding number of **linear segments** might be useful. This means add kinks to generator and possibly fine tune models on physical simulations (Falaise).
## Technical stuff
The way to add tensorflow into binaries into [TKEvent](https://github.com/TomasKrizak/TKEvent) was found. It will only reqire c api for TensorFlow which can be downloaded from TensorFlow webside and links dynamically to analysation code. It might be even possible to use this solution within root macros (i.e. running code via `root <filename>.cxx`). For more information, see "Working with real data (future-proof)" section.

**From now, this section is only about results on generator**
## Associated calorimeter hits
Performance depends on number of tracks in event. For one track, we have 98% accuracy. For more tracks, it is multilabel classification problem, which is much more harder to analyse and measure performance for. However, we can say that multilabel classification approach from Meta Research (Softmax on one hot encoding divided by number of tracks) performs significantly better than classical approach with BinaryCrossentropy. See [Results](#results) section. However associated calohit without clustering is probably useless and if we have working clustering, we can then us single label classifcation on one track so there is nothing groundbreaking here.
## Clustering
Three strategies proposed:
 1. Approach by Matteo (basically SegNET architecture - see resources in the end of this document) enhanced by Generative Adversarial Networks.
     * 
 2. Approach by Matteo enhanced by some kind of attention mechanism. It has been shown (for example on LLMs) that attention models sequences and relations between features well. It can be then trained using GANs.
 3. Train simple autoencoder. Then, disconnect decoder and use only encoder. The clustering/image segmentation will be done within latent space (output of encoder). It means that we will generate latent representation of event (r1), then we will generate latent representation of event without one track (r2) and train model to go from (r1) to (r2). if we want to see clustered event, we can push the modified image latent representation into decoder.
     * Cannot find simple resources about image segmentation within latent space (only really complicated modern foundation models which are definitely overkill for SuperNEMO tracker).
     * Will be beneficial only if the latent space is small.
     * Some results from fitting givethe idea that this will not work (see next subsection).
     * Some of these problems might be resolved by using **Variational autoencoder**.
 4. Model (basically one layer with convolutional filters) with two channels as input. One channel wil be the actual event and the second will be the track that we are clustering. 
### Generative Adversarial Networks
In literature, the approach of learning similarity metrics in GAN manner is called (V)AE/GAN depending on whether autoencoder is variational. See for example [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/pdf/1512.09300.pdf).

Testing clustering on events 
Convolutional autoencoder with output neuron trained on classification tasks:
 1. is significant number of hits in the track no. 2 missing?
 2. is significant number of hits in the track no. 1 present?
 3. and possibly any other criterion where the generator would be failing.
### Variational Autoencoders
Autoencoders do not by default produce nice almost everywhere differentiable distribution in latent space, therefore Variational autoencoders are commonly used. See [Resources](#resources) section.
## Fitting
 * Trying to give hint to [TKEvent](https://github.com/TomasKrizak/TKEvent), where it should search for solution (angle - 5 segments, position on foil - 10 segments)
 * Mixed results: For one track we have approximately 70% accuracy (see [Results](#results) section). Then it falls for events with more tracks.
# Help needed from collaboration
 * Information about kinks.
# Software
Information in this section are mostly for CCLyon in2p3 computation cluster.
## Required software
Almost everything runs ot top of `python3`. On CCLyon use `python` sourced with `root` via
1. `ccenv root 6.22.06` - loads `python 3.8.6` (**does not work now**)
2. since July 12 2023 `module add Analysis/root/6.22.06-fix01` - loads `python 3.9.1` (**currently, this is the way to go**)

This software should be installed in python or Anaconda environment (python environment is prefered since it can access  both sourced root package and all gpu related software directly, however it is still possible to make it work with Anaconda)
 * `root` - Root is not needed to be explicitly installed in python or Anaconda environment, any sourced Root on CCLyon should work - minimum tested verion 6.22.06 (since July 12 2023 6.22.06-fix01 on CCLyon). **PyROOT is required.**
 * `cudatoolkit`, `cudnn` - Should be already installed on CCLyon 
 * `tensorflow` - (ideally 2.13, older version produce some random bug with invalid version of openSSL build)
 * `keras` - Should be part of `tensorflow`
 * `keras-tuner` - hyperparameter tuning
 * `numpy`
 * `maplotlib`, `seaborn` - plotting
 * `scikit-learn` - some helper functions
 * `pydot`, `graphviz` - drawing models
 * `argparse`
 
Optional:
 * `tensorrt` - really useful, makes inference and training of models faster on CPU. It can be also installed in two steps, first install `nvidia-pyindex` and then `nvidia-tensorrt`. **To use it succesfully within scripts, you should import it before tensorflow!**
 * `tensorboard` - Should be part of `tensorflow`
 * `tensorboard_plugin_profile` - profiling
 * `nvidia-pyindex`, `nvidia-tensorrt` - For TensorRT support
 * `nvidia-smi` -  For checking usage and available memory on NVIDIA V100 GPU (on CCLyon)

## Running scripts (on CCLyon in2p3 cluster)
Example is at `example_exec.sh`. Run it with `sbatch --mem=... -n 1 -t ... gres=gpu:v100:N example_exec.sh` if you have access to GPU, where `N` is number of GPUs you want to use (currently CCLyon does not allow me to use more than three of them) Otherwise, leave out `gres` option.

Scripts can use two strategies. To use only one GPU use option `--OneDeviceStrategy "/gpu:0"`. If you want to use more GPUs, use for example `--MirroredStrategy "/gpu:0" "/gpu:1" "/gpu:2"`. For some reason, I was never able to use more than 3 GPUs on CClyon.

If you start job from bash instance with some packages, modules or virtual environment loaded, you should unload them/deactivate them (use `module purge --force`). Best way is to start from fresh bash instance.
## Workflow overview
1. source `root` (and `python`) - **currently use `module add Analysis/root/6.22.06-fix01`**
2. create python virtual environment (if not done yet) 
3. install [packages](#required-software) (if not done yet)
4. load python virtual environment (or add `#! <path-to-your-python-bin-in-envorinment>` to first line of your script)
## Working with real data (temporary solution)
We test models on real data and compared them with [TKEvent](https://github.com/TomasKrizak/TKEvent). Unfortunately, it is not possible to open `root` files produced by [TKEvent](https://github.com/TomasKrizak/TKEvent) library from python with the same library sourced since this library might be built with different version of python and libstdc++. Fortunately, workaround exists. We need to download and build two versions of [TKEvent](https://github.com/TomasKrizak/TKEvent). First version will be built in the manner described in [TKEvent](https://github.com/TomasKrizak/TKEvent) README.md. The second library shoudl be build (we ignore the `red_to_tk` target) with following steps:

1. `module add ROOT` where `ROOT` is version of `root` library used by `tensorflow` (**currently `module add Analysis/root/6.22.06-fix01`**)
2. `TKEvent/TKEvent/install.sh` to build library 

Now, we can use `red_to_tk` from the first library to obtain root file with `TKEvent` objects and open this root file with the second version of `TKEvent` library.
## Working with real data (future-proof)
If the collaboration will want to use keras models inside software, the best way is probably to use [cppflow](https://github.com/serizba/cppflow) . It is single header c++ library for acessing TensoFlow C api. This means that we will not have to build TensorFlow from source and we should not be restricted by root/python/gcc/libstdc++ version nor calling conventions. 
## Local modules
Not to be restricted by the organization structure of our folders, we use this script to load, register and return local modules.
```
def import_arbitrary_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module
```
## Issues
 1. `sbatch` and `tensorflow` sometimes fail to initialize libraries (mainly to source python from virtual environment or root) - start the script again ideally from new bash instance without any modules nor virtual environment loaded.
 2. `tensorflow` sometimes runs out of memory - Don't use checkpoints for `tensorboard`. Another cause of this problem might be training more models in one process, we can solve this by `keras.backend.clear_session()`. If this error occurs after several hours of program execution, check out function `tf.config.experimental.set_memory_growth`. 
 3. TensorFlow 2.13 distributed training fail - https://github.com/tensorflow/tensorflow/issues/61314
 4. Sometimes, there are strange errors regarding ranks of tensors while using custom training loop in `gan.py` - looks like really old still unresolved bug inside core tensorflow library. However, the workaround is to pass only one channel into CNN architecture and concat them with `tf.keras.layers.Concatenate`
 5. `numpy` ocasionally stops working (maybe connected to [this](https://github.com/pypa/pip/issues/9542) issue), failing to import (`numpy.core._multiarray_umath` or other parts of `numpy` library). Is is accompanied by message:
 ```
IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.
 ``` 
 Simplest solution is to create new environment and install all libraries inside that.
# Description of files
 * `lib.py` -  small library with some functions that are reused across this project 
 * `number_of_tracks_classification.py`
 * `combined.py` - Script for constructing and training model consisting from `top`, `side` and `front` models.
 * `vae.py` - Trains VAE models
 * `number_of_tracks_classification3D.py` - TODO: Classification using Conv3D
 * `plot_confusion.py` - Script helping analyze badly classified events
 * `clustering_one.py`- example of custom trainig loop for GAN autoencoders used for clustering
 * `multilabel_analysis.py` - script for analysing various aspect of multilabel classifier used for associated calorimeter hit detection
 * `gan.py` - 
 * `testing_clustering.py` - code for testing tracker hit clustering algorithms
 * `self_attention.py` - primitive self-attention model for convolutional autoencoder within GAN
 * `capsule_lib.py` - basic parts of CapsNET architecture
   * For more information about CapsNET and capsules, see [Resources](#resources)
   * `class RoutingByAgreement` implements routing by agreement procedure proposed by Hinton using `tf.while_loop`
   * `class PrimaryCapsule` - implements the first capsule layer in CapsNET architecture
   * `class SecondaryCapsule` implements the second capsule layer in CapsNET architecture 
## `gan_tests`
Testing whether my GAN algorithm works. Replicating https://www.tensorflow.org/tutorials/generative/dcgan. One important problem found: when missing batch normalizatio inside generator architecture, gan fails to reprodudce digits. Possible solutions for modelling longer shapes such a
tracks might be self-attention (really want this in architecture, massively beneficial for any ML task and implemented in CapsNET). See short article in [Resources](#convolutional-neural-networks-autoencoders-and-gans).
## `architectures`
 * `top.py`
 * `top_big.py`
 * `side.py`
 * `front.py`
 * `combined.py` - code generating combined model
 * `top_associated_calorimeter.py`
 * `front_associated_calorimeter.py`
 * `side_associated_calorimeter.py`
 * `generator.py` - Generator used in GAN architecture
 * `discriminator.py` - Discriminator used in GAN architecture
 * `matteo_with_skip_connections.py` - Autoencoder for clustering proposed and tested by Matteo (don't know exactly what the results are and if it was working at all). In fact, it is modified SegNET (see [Resources](#resources) section).
  * `matteo_without_skip_connections.py` - The same as above but the skip connections are removed. This should not work for clustering, but it will work as autoencoder.

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
 * `matteo_with_skip` - Autoencoder based on Matteos architecture (`my_generator`)
 * `matteo_without_skip` - Autoencoder based on Matteos architecture (`my_generator`)
 * `clustering_matteo_with_skip_connections` - Learned to remove track on left from events with two tracks (`my_generator`)
 * `clustering_matteo_without_skip_connections` - Learned to remove track on left from events with two tracks (`my_generator`)
## `enhanced_fitting`
First attempts to use ML to help [TKEvent](https://github.com/TomasKrizak/TKEvent).
 * `TKEvent` - slightly modified [TKEvent](https://github.com/TomasKrizak/TKEvent) library.
 * `fit_one_iteratively.py` - uses ml to predict number of tracks and fits one track, removes associated tracker hits from event and repeats until the predicted tracks are fitted
 * `special_events.py` - can modify events and inspect differences between number of predicted tracks before modificatio and after
## `VAE` 
Helper files for Variational Autoencoder.
 * `decoders.py` - decoders for VAEs
 * `encoders.py` - encoders for VAEs
 * `lib.py` - VAE architecture and layer for Reparametrization trick
 * `my_dataset_with_hint.py` - implementation of tf.datasets workflow for VAE and my_generator
## `Matteo`
Aproach to clustering done by Matteo. Sometimes works really well, sometimes really badly.
# Results 
## SN-IEGenerator, my_generator
 * [Confusion matrix for combined model (SN-IEGenerator)](./ImagesAndDocuments/combined.pdf)
 * [Confusion matrix for top model (my_generator)](./ImagesAndDocuments/top_model_my_generator_confusion_matrix.pdf)
 * [Confusion matrix for side model (my_generator)](./ImagesAndDocuments/side_model_my_generator_confusion_matrix.pdf)
 * [Confusion matrix for front model (my_generator)](./ImagesAndDocuments/front_model_my_generator_confusion_matrix.pdf)
 * [Confusion matrix for combined model (my_generator)](./ImagesAndDocuments/combined_model_my_generator_confusion_matrix.pdf)
 * [Confusion matrix for prediction of angle from top view (my_generator)](./ImagesAndDocuments/angle_1_confusion.pdf)
 * [Comparison of accuracy for associated calorimeter (classical sigmoid approach and softmax approach proposed by Meta Research for multi-label classification tasks)](./ImagesAndDocuments/multilabel.pdf) - please not that multi-label classification is complex task, so these results **require further analysis**! 
 * [Clustering approach by Matteo - learned on my_generator with events with 2 tracks to remove left track - sometimes works, sometimes doesnt](./ImagesAndDocuments/matteo_testing)
   * To see this well cat this file into terminal.
## Real data
 * [Prediction of number of tracks on real data](./ImagesAndDocuments/top_model_classification)
   * only top view
   * run 974
   * Top model predicted number of tracks and [TKEvent](https://github.com/TomasKrizak/TKEvent) tried to fit this number of tracks/linear segments.
   * isolated tracker hits filtered
# Resources
## Convolutional neural networks, autoencoders and GANs
 * [Multi-label image classification](https://towardsdatascience.com/multi-label-image-classification-with-neural-network-keras-ddc1ab1afede)
 * [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan)
 * [Building a simple Generative Adversarial Network (GAN) using TensorFlow](https://blog.paperspace.com/implementing-gans-in-tensorflow/)
 * [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf)
 * Variational autoencoders [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
    * anbles nicer distribution in latent space, which then can be reused for other ML algorithms or used as generator
 * [Adversarial autoencoders](https://arxiv.org/pdf/1511.05644.pdf)
    * connect distriminator to latent space and force encoder to produce latent space mimicking the prefered distribution
    * next diccusses semi-superwised learning and unsuperwised clustering (clustering of images, not image segmentation what we want to achieve)
  * [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/pdf/1512.09300.pdf)
    * Summarizes AE, VAE, VAE/GAN approach (my approach for clustering tracker hits)
  * [Advanced GANs - Exploring Normalization Techniques for GAN training: Self-Attention and Spectral Norm](https://sthalles.github.io/advanced_gans/)
  * [Attention + convolution #1](https://arxiv.org/abs/1904.09925), [Attention + convolution #2](https://medium.com/mlearning-ai/self-attention-in-convolutional-neural-networks-172d947afc00), [Attention + convolution #3](https://medium.com/ai-salon/understanding-deep-self-attention-mechanism-in-convolution-neural-networks-e8f9c01cb251)
  * [Original CBAM paper](https://arxiv.org/pdf/1807.06521.pdf)
## Image Segmentation (clustering tracker hits)
 * [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/pdf/2001.05566.pdf)
 * [SegNET](https://arxiv.org/pdf/1511.00561.pdf)
## CapsNET
 * [Series of articles on Capsule architecture](https://pechyonkin.me/capsules-1/)
 * [Preprint on routing algorithm (Capsule architecture)](https://arxiv.org/abs/1710.09829)
 * [Tensorflow implementation 1](https://www.kaggle.com/code/giovanimachado/capsnet-tensorflow-implementation)
 * [Tensorflow implementation 2](https://towardsdatascience.com/implementing-capsule-network-in-tensorflow-11e4cca5ecae)
 * [Matrix Capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb)
## Tensorflow
 * [Dense layer in keras (how to build custom keras model)](https://github.com/keras-team/keras/blob/v2.12.0/keras/layers/core/dense.py)
 * [Tricks for custom layers](https://oliver-k-ernst.medium.com/a-cheat-sheet-for-custom-tensorflow-layers-and-models-aa465df2bc8b)
## Classification
 * [Meta, Instagram: Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/pdf/1805.00932.pdf) - [Discussion #1](https://stats.stackexchange.com/questions/433867/why-is-softmax-considered-counter-intuitive-for-multi-label-classification), [Discussion #2](https://stackoverflow.com/questions/66990074/using-softmax-for-multilabel-classification-as-per-facebook-paper)
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
 * [Overview of conditional random fields](https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541)
 * [Transformer position as dynamical model](https://assets.amazon.science/f0/32/ff7d9669492bbe2dedb8ee3cb3e5/learning-to-encode-position-for-transformer-with-continuous-dunamical-model.pdf)
 * [Neural Ordinary Differential Equations](https://papers.nips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html)
 * [Learning Jacobian Trace of Ordinary Differential Equation](https://arxiv.org/pdf/1810.01367)
 * [INVERTIBLE SURROGATE MODELS FOR LASERWAKEFIELD ACCELERATION](https://simdl.github.io/files/32.pdf)

