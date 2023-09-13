# VPRTempo - Temporally encoded spiking neural network for visual place recognition
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/QVPR/VPRTempo.svg?style=flat-square)](https://github.com/QVPR/VPRTempo/stargazers)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)

This repository contains code for VPRTempo, a spiking neural network that uses temporally encoding to perform visual place recognition tasks. The network is based off of [BLiTNet](https://arxiv.org/pdf/2208.01204.pdf) and adapted to the [VPRSNN](https://github.com/QVPR/VPRSNN) framework. 

<p style="width: 50%; display: block; margin-left: auto; margin-right: auto">
  <img src="./assets/github_image.png" alt="VPRTempo method diagram"/>
</p>

## License & Citation
This repository is licensed under the [MIT License](./LICENSE)

If you use our code, please cite the following paper below:

## Installation and setup
We recommend installing dependencies for VPRTempo with [Mambaforge](https://mamba.readthedocs.io/en/latest/installation.html), however `conda` may also be used. VPRTempo uses [PyTorch](https://pytorch.org/) with the capability for [CUDA](https://developer.nvidia.com/cuda-toolkit) GPU acceleration. Follow the installation instructions based on your operating system and hardware specifications.

### Windows & Linux
#### CUDA enabled installation
Use conda/mamba to create a new environment and install Python, CUDA tools, and dependencies.

```console
conda create -n vprtempo -c pytorch -c nvidia python torchvision torchaudio pytorch-cuda=11.7 cudatools opencv matplotlib
```
> **Note**
> Install the version of PyTorch-CUDA that is compatible with your graphics card, see [Start Locally | PyTorch](https://pytorch.org/get-started/locally/) for more details.

When running the code, it will automatically detect and print the device being used for the network. If your CUDA install was successful, you should expect an output like:

```python
> CUDA available: True
> Current device is: NVIDIA GeForce RTX 2080
```
If CUDA was not installed correctly, the output will be:
```python
> CUDA available: False
> Current device is: CPU
```

#### CPU only
To install using the CPU only, simply install Python + dependencies.
```console
conda create -n vprtempo python pytorch torchvision torchaudio cpuonly opencv matplotlib -c pytorch
```
### MacOS
CUDA acceleration is not available on MacOS and the network will only use the CPU, so simply just need to install Python + dependencies.
```console
conda create -n vprtempo -c conda-forge python opencv matplotlib -c pytorch pytorch::pytorch torchvision torchaudio
```

### Get the repository
Activate the environment & download the Github repository
```console
conda activate vprtempo
git clone https://github.com/QVPR/VPRTempo.git
cd ~/VPRTempo
```

## Datasets

### Nordland
VPRTempo was developed and tested using the [Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset). This software will work for either the full-resolution or down-sampled datasets, however our paper details the full-resolution datasets. 

To simplify first usage, we have set the defaults in `VPRTempo.py` to train and test on a small subset of Nordland data. We recommend [downloading Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset) and using the `./src/nordland.py` script to unzip and organize the images into the correct file and naming structure.

### Custom datasets
In general, data should be organised in the following way in order to train the network on multiple traversals of the same location.

```
--dataset
  |--training
  |  |--traversal_1
  |  |--traversal_2
  |
  |--testing
  |  |--test_traversal
```
Speicfy the datapaths by altering `self.trainingPath` and `self.testPath` in `VPRTempo.py`. You can specify which traversals you want to train and test on by also altering `self.locations` and `self.test_location`. In the case above it would be the following; 

```python
self.trainingPath = '<path_to_data>/training/
self.testPath = '<path_to_data>/testing/

self.locations = ["traversal_1","traversal_2"]
self.test_location = "test_traversal"
```

Image names for the same locations across traversals (training and testing) must be the same as they are imported based on a `.txt` file. 

## Usage
Both the training and testing is handled by the `VPRTempo.py` script. Initial installs do not contain any pre-defined networks and will need to be trained prior to use.
### Pre-requisites
* Training and testing data is organized as above (see **Datasets** on how to set up the Nordland or custom datasets)
* The VPRTempo `conda` environment has been activated

Once these two things have been setup, run `VPRTempo.py` to train and test your first network with the default settings. 

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/QVPR/VPRTempo/issues).
