# VPRTempo - Temporally encoded spiking neural network for visual place recognition
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/QVPR/VPRTempo.svg?style=flat-square)](https://github.com/QVPR/VPRTempo/stargazers)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)

This repository contains code for VPRTempo, a spiking neural network that uses temporally encoding to perform visual place recognition tasks. The network is based off of [BLiTNet](https://arxiv.org/pdf/2208.01204.pdf) and adapted to the [VPRSNN](https://github.com/QVPR/VPRSNN) framework. 

## License & Citation
This repository is licensed under the [MIT License](./LICENSE)

If you use our code, please cite the following paper below:

## Installation and setup
We recommend installing dependencies for VPRTempo with [Mambaforge](https://mamba.readthedocs.io/en/latest/installation.html), however `conda` may also be used. VPRTempo uses [PyTorch](https://pytorch.org/) with the capability for [CUDA](https://developer.nvidia.com/cuda-toolkit) GPU acceleration. Follow the installation instructions based on your operating system and hardware specifications.

### Windows & Linux
#### CUDA enabled installation
First, download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and follow the installation instructions for your hardware specifications.

Once installed, create the new environment and install Python + dependencies.

```console
mamba create -n vprtempo -c pytorch -c nvidia python torchvision torchaudio pytorch-cuda=11.7 opencv matplotlib halo
```
> **Note**
> Install the version of PyTorch-CUDA that is compatible with your graphics card, see [Start Locally | PyTorch](https://pytorch.org/get-started/locally/) for more details.

When running the code, it will automatically detect and print the device being used for the network. If your CUDA install was successful, you should expect an output like:

```python
import torch

print('CUDA available: '+str(torch.cuda.is_available()))
        if torch.cuda.is_available() == True:
            current_device = torch.cuda.current_device()
            print('Current device is: '+str(torch.cuda.get_device_name(current_device)))
        else:
            print('Current device is: CPU')

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
mamba create -n vprtempo python pytorch torchvision torchaudio cpuonly opencv matplotlib halo -c pytorch
```
### MacOS
CUDA acceleration is not available on MacOS and the network will only use the CPU, so simply just need to install Python + dependencies.
```console
mamba create -n vprtempo -c conda-forge python opencv matplotlib halo -c pytorch pytorch::pytorch torchvision torchaudio
```

### Get the repository
Activate the environment & download the Github repository
```console
conda activate vprtempo
git clone https://github.com/QVPR/VPRTempo.git
cd ~/VPRTempo
```

## Datasets
VPRTempo was developed and tested using the [Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset) dataset, an approximately 729km traverse of the Nordland Railway in Norway recorded over the 4 seasons - Spring, Fall, Summer, & Winter. This software will work for either the full-resolution or down-sampled datasets, however our paper details the full-resolution datasets. 

## Usage
Both the training and testing is handled by the `VPRTempo.py` script. Initial installs do not contain any pre-defined networks and will need to be trained prior to use. There are only two pre-requisites for running the network.
### Pre-requisites
* The Nordland dataset is downloaded wih the Spring and Fall images from sections 1 & 2 placed into a their own folders (training data), and the Summer images similarly placed in another separate folder (testing data).
* Edit lines 61 & 62 of `VPRTempo.py` to your local training and testing folders

Once these two things have been setup, run `VPRTempo.py` to train and test your first network with the default settings.

## Training and testing benchmarks
System benchmarks are based off the following network settings:
| Input neurons | Feature neurons | Output neurons | Expert modules | Total network size | Training images | Epochs | Testing images |
|     :---:     |      :---:      |     :---:      |     :---:      |     :----:         |   :----:        | :---:  |    :---:       |
|      784      |      784        |       25       |      200       |      126,851,200   |   10,000        |   5    |    1,000       |

Data presented as the average of 5 independent training and testing ± the standard deviation. 

| Device | Training time (s) | Query speed (Hz) | Training PRG (%) | Query PRG (%)|
| :---:  |   :---:           |   :---:          |          :---:   |    :---:     |
|NVIDIA GeForce RTX 2080|32.17 ± 0.342|192.73 ± 17.07| - | - |
|Intel® Core™ i7-9700K CPU @ 3.60GHz|1274.11 ± 287.14|16.17 ± 4.23|2.52|8.39|
|Apple M1 Pro|
> **Performance relative to GPU (PRG)**

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/QVPR/VPRTempo/issues).
