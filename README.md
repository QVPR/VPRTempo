# VPRTempo - Temporally encoded spiking neural network for visual place recognition
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/QVPR/VPRTempo.svg?style=flat-square)](https://github.com/QVPR/VPRTempo/stargazers)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)

This repository contains code for VPRTempo, a neural network that uses temporally encoded spikes to perform visual place recognition tasks. The temporal spiking network is based off of [BLiTNet](https://arxiv.org/pdf/2208.01204.pdf) and adapted to [VPRSNN](https://github.com/QVPR/VPRSNN). 

## License
This repository is licensed under the [MIT License](./LICENSE)

## Installation and setup
We recommend using [Mambaforge](https://mamba.readthedocs.io/en/latest/installation.html) to install all package dependicies. VPRTempo uses [PyTorch](https://pytorch.org/) and has the capability for [CUDA](https://developer.nvidia.com/cuda-toolkit) GPU acceleration. Follow the installation instructions based on your operating system.
> **Note**
> CUDA acceleration is not available on MacOS and the network will only use the CPU.
### Windows & Linux
#### Option 1 - CUDA enabled GPU acceleration
```console
$ mamba create -n vprtempo -c conda-forge python opencv matplotlib alive-progress -c pytorch -c nvidia torchvision torchaudio pytorch-cuda=11.7
```
#### Option 2 - CPU only
```console
$ mamba create -n vprtempo -c conda-forge python opencv matplotlib alive-progress -c pytorch pytorch torchvision torchaudio cpuonly
```
Once the environment has been created, activate it and download the repository.
```console
$ mamba activate vprtempo
$ git clone https://github.com/QVPR/VPRTempo.git
$ cd ~/VPRTempo
```
### MacOS
```console
$ mamba create -n vprtempo -c conda-forge python opencv matplotlib alive-progress -c pytorch pytorch::pytorch torchvision torchaudio
```
Once the environment has been created, activate it and download the repository.
```console
$ mamba activate vprtempo
$ git clone https://github.com/QVPR/VPRTempo.git
$ cd ~/VPRTempo
```
### Datasets
VPRTempo was developed and tested using the [Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset) dataset. Details on training and testing images are described below.

## Usage
Both the training and testing is handled by the `VPRTempo.py` script. Initial installs do not contain any pre-defined networks and will need to be trained prior to use.
### Pre-requisites
* The Nordland dataset is downloaded wih the Spring and Fall images placed into a separate folder (training data), and the Summer images placed in another separate folder (testing data).
  * The `data_mover.py` tool has been provided to conveniently do this for you and rename the files appropriately, this can be found in the `./src` folder.
  * Edit lines 50 & 51 of `VPRTempo.py` to your local training and testing folders
### First run
