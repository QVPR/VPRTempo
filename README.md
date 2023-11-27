# VPRTempo - Temporally encoded spiking neural network for visual place recognition
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/QVPR/VPRTempo.svg?style=flat-square)](https://github.com/QVPR/VPRTempo/stargazers)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)
![GitHub repo size](https://img.shields.io/github/repo-size/QVPR/VPRTempo.svg?style=flat-square)

This repository contains code for VPRTempo, a spiking neural network that uses temporally encoding to perform visual place recognition tasks. The network is based off of [BLiTNet](https://arxiv.org/pdf/2208.01204.pdf) and adapted to the [VPRSNN](https://github.com/QVPR/VPRSNN) framework. 

<p style="width: 50%; display: block; margin-left: auto; margin-right: auto">
  <img src="./assets/github_image.png" alt="VPRTempo method diagram"/>
</p>

## License & Citation
This repository is licensed under the [MIT License](./LICENSE)

If you use our code, please cite the following [paper](https://arxiv.org/abs/2309.10225):
```
@misc{hines2023vprtempo,
      title={VPRTempo: A Fast Temporally Encoded Spiking Neural Network for Visual Place Recognition}, 
      author={Adam D. Hines and Peter G. Stratton and Michael Milford and Tobias Fischer},
      year={2023},
      eprint={2309.10225},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
## Installation and setup
VPRTempo uses [PyTorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-toolkit) GPU acceleration. Follow the installation instructions based on your operating system and hardware specifications. MacOS has no compatibly with CUDA.
### Get the repository
Download the Github repository.
```console
git clone https://github.com/QVPR/VPRTempo.git
cd ~/VPRTempo
```
Once downloaded, please install the required dependencies to run the network through one of the following options:

### Option 1: Pip install
Dependencies for VPRTempo can downloaded from our PyPi package.

```console
# For Windows/Linux systems
!pip install vprtempo

# For MacOS
!pip install vprtempomacos
```

### Option 2: Local requirements install
Dependencies can be installed either through our provided `requirements.txt` files.

```console
# For Windows/Linux
!pip install -r requirements.txt

# For MacOS
!pip instal -r requirements_macos.txt
```
### Option 3: Conda install
>**:heavy_exclamation_mark: Recommended:**
> Use [Mambaforge](https://mamba.readthedocs.io/en/latest/installation.html) instead of conda.

```console
# Windows/Linux - CUDA enabled
conda create -n vprtempo -c pytorch -c nvidia python torchvision torchaudio pytorch-cuda=11.7 cudatoolkit prettytable tqdm numpy pandas scikit-learn

# Windows/Linux - CPU only
conda create -n vprtempo python pytorch torchvision torchaudio cpuonly prettytable tqdm numpy pandas scikit-learn -c pytorch

# MacOS
conda create -n vprtempo -c conda-forge python prettytable tqdm numpy pandas scikit-learn -c pytorch pytorch::pytorch torchvision torchaudio
```

## Datasets
VPRTempo was developed to be simple to train and test a variety of datasets. Please see the information below about running a test with the Nordland traversal dataset and how to organize custom datasets.

### Nordland
VPRTempo was developed and tested using the [Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset) traversal dataset. This software will work for either the full-resolution or down-sampled datasets, however our paper details the full-resolution datasets. 

To simplify first usage, we have set the defaults in `VPRTempo.py` to train and test on a small subset of Nordland data. We recommend [downloading Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset) and using the `./src/nordland.py` script to unzip and organize the images into the correct file and naming structure.

### Custom datasets
In general, data should be organised in the `./dataset` folder in the following way in order to train the network on multiple traversals of the same location.

```
--dataset
  |--training
  |  |--traversal_1
  |  |--traversal_2
  |
  |--testing
  |  |--test_traversal
```
If you wish to specify a different directory where data is stored, modify the `--data_dir` default argument in `main.py`. Similarly, if you wish to train/query different traversals modify `--database_dirs` and `--query_dir` in `main.py` accordingly.

## Usage
Both the training and testing is handled by the `VPRTempo.py` script. Initial installs do not contain any pre-defined networks and will need to be trained prior to use.
### Pre-requisites
* Training and testing data is organized as above (see **Datasets** on how to set up the Nordland or custom datasets)
* The VPRTempo `conda` environment has been activated

Once these two things have been setup, run `VPRTempo.py` to train and test your first network with the default settings. 

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/QVPR/VPRTempo/issues).
