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
Use conda/mamba to create a new environment and install Python, CUDA tools, and dependencies.

```console
conda create -n vprtempo -c pytorch -c nvidia python torchvision torchaudio pytorch-cuda=11.7 cudatools opencv matplotlib
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
VPRTempo was developed and tested using the [Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset) dataset, an approximately 729km traverse of the Nordland Railway in Norway recorded over the 4 seasons - Spring, Fall, Summer, & Winter. This software will work for either the full-resolution or down-sampled datasets, however our paper details the full-resolution datasets. 

In general, data should be organised in the following way in order to train the network on multiple traversals of the same location.

```
--Dataset
  |--Training
  |  |--Traversal_1
  |  |--Traversal_2
  |  |-- ...
  |  |--Traversal_n
  |
  |--Testing
  |  |--Test_traversal
```
Speicfy the datapaths by altering `self.trainingPath` and `self.testPath` in lines 60 and 61 of `VPRTempo.py`. You can specify which traversals you want to train and test on by also altering `self.locations` and `self.test_location` in lines 66 and 67. In the case above it would be the following; 

```python
60 self.trainingPath = '<path_to_data>/Training/
61 self.testPath = '<path_to_data>/Testing/

66 self.locations = ["Traversal_1,Traversal_2"]
67 self.test_location = "Test_traversal"
```

Image names for the same locations across traversals (training and testing) must be the same as they are imported based on a `.txt` file. We provide an easy tool in `./src/generate_names.py` that will go through and make sure all the names are the same and generate a `.txt` file of image names and store it in the main folder.

## Usage
Both the training and testing is handled by the `VPRTempo.py` script. Initial installs do not contain any pre-defined networks and will need to be trained prior to use. There are only two pre-requisites for running the network.
### Pre-requisites
* The Nordland dataset is downloaded wih the Spring and Fall images from sections 1 & 2 placed into a their own folders (training data), and the Summer images similarly placed in another separate folder (testing data).
* Edit lines 61 & 62 of `VPRTempo.py` to your local training and testing folders

Once these two things have been setup, run `VPRTempo.py` to train and test your first network with the default settings. 

## Training and testing benchmarks
System benchmarks are based off the following network settings:
| Input neurons | Feature neurons | Output neurons | Expert modules | Total network size (no. neurons) | Training images | Epochs | Testing images |
|     :---:     |      :---:      |     :---:      |     :---:      |     :----:         |   :----:        | :---:  |    :---:       |
|      784      |      1568       |       165       |      20       |      50,340   |   6,600       |   4    |    3,300      |

Data presented as the average of 5 independent training and testing ± the standard deviation. 

| Device | Training time (s) | Query speed (Hz) | Training PRG (%) | Query PRG (%)|
| :---:  |   :---:           |   :---:          |          :---:   |    :---:     |
|NVIDIA GeForce RTX 2080|46.96 ± 0.80|137.9 ± 5.48| - | - |
|Intel® Core™ i7-9700K CPU @ 3.60GHz| ± | ± |||

> **Performance relative to GPU (PRG)**

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/QVPR/VPRTempo/issues).
