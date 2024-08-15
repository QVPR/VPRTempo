# VPRTempo - A Temporally Encoded Spiking Neural Network for Visual Place Recognition
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)
[![stars](https://img.shields.io/github/stars/QVPR/VPRTempo.svg?style=flat-square)](https://github.com/QVPR/VPRTempo/stargazers)
[![Downloads](https://static.pepy.tech/badge/vprtempo)](https://pepy.tech/project/vprtempo)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/vprtempo.svg)](https://anaconda.org/conda-forge/vprtempo)
![PyPI - Version](https://img.shields.io/pypi/v/VPRTempo)

This repository contains code for [VPRTempo](https://vprtempo.github.io), a spiking neural network that uses temporally encoding to perform visual place recognition tasks. The network is based off of [BLiTNet](https://arxiv.org/pdf/2208.01204.pdf) and adapted to the [VPRSNN](https://github.com/QVPR/VPRSNN) framework. 

<p style="width: 50%; display: block; margin-left: auto; margin-right: auto">
  <img src="./assets/vprtempo_example.gif" alt="VPRTempo method diagram"/>
</p>

VPRTempo is built on a [torch.nn](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) framework and employs custom learning rules based on the temporal codes of spikes in order to train layer weights. 

In this repository, we provide two networks: 
  - `VPRTempo`: Our base network architecture to perform visual place recognition (fp32)
  - `VPRTempoQuant`: A modified base network with [Quantization Aware Training (QAT)](https://pytorch.org/docs/stable/quantization.html) enabled (int8)

To use VPRTempo, please follow the instructions below for installation and usage.

## :star: Update v1.1: What's new?
  - Full integration of VPRTempo into torch.nn architecture
  - Quantization Aware Training (QAT) enabled to train weights in int8 space
  - Addition of tutorials in Jupyter Notebooks to learn how to use VPRTempo as well as explain the computational logic
  - Simplification of weight operations, reducing to a single weight tensor - allowing positive and negative connections to change sign during training
  - Easier dependency installation with PyPi/pip and conda
  - And more!

## License & Citation
This repository is licensed under the [MIT License](./LICENSE) 

If you use our code, please cite our IEEE ICRA [paper](https://ieeexplore.ieee.org/document/10610918):
```
@inproceedings{hines2024vprtempo,
      title={VPRTempo: A Fast Temporally Encoded Spiking Neural Network for Visual Place Recognition}, 
      author={Adam D. Hines and Peter G. Stratton and Michael Milford and Tobias Fischer},
      year={2024},
      pages={10200-10207},
      booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}     
}
```
## Installation and setup
VPRTempo uses [PyTorch](https://pytorch.org/) with the capability for [CUDA](https://developer.nvidia.com/cuda-toolkit) acceleration. Please use one of the following options below to install the required dependencies, and if desired follow the instructions to install CUDA for your hardware and operating system.
### Get the repository
Download the Github repository.
```console
git clone https://github.com/QVPR/VPRTempo.git
cd ~/VPRTempo
```
Once downloaded, please install the required dependencies to run the network through one of the following options:

### Option 1: Pip install
Dependencies for VPRTempo can downloaded from our [PyPi package](https://pypi.org/project/VPRTempo/).

```python
pip install vprtempo
```
If you wish to enable CUDA, please follow the instructions on the [PyTorch - Get Started](https://pytorch.org/get-started/locally/) page to install the required software versions for your hardware and operating system.

### Option 2: Local requirements install
Dependencies can be installed either through our provided `requirements.txt` files.

```python
pip install -r requirements.txt
```
As above, if you wish to install CUDA please visit [PyTorch - Get Started](https://pytorch.org/get-started/locally/).
### Option 3: Conda install
>**:heavy_exclamation_mark: Recommended:**
> Use [Mambaforge](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) instead of conda.

Requirements for VPRTempo may be installed using our [conda-forge package](https://anaconda.org/conda-forge/vprtempo).

```console
# Linux/OS X
conda create -n vprtempo -c conda-forge vprtempo

# Linux CUDA enabled
conda create -n vprtempo -c conda-forge -c pytorch -c nvidia vprtempo pytorch-cuda cudatoolkit

# Windows
conda create -n vprtempo -c pytorch python pytorch torchvision torchaudio cpuonly prettytable tqdm numpy pandas scikit-learn

# Windows CUDA enabled
conda create -n vprtempo -c pytorch -c nvidia python torchvision torchaudio pytorch-cuda=11.7 cudatoolkit prettytable tqdm numpy pandas scikit-learn

```

## Datasets
VPRTempo was developed to be simple to train and test a variety of datasets. Please see the information below about running a test with the Nordland and Oxford RobotCar datasets and how to organize custom datasets.

Please note that while we trained 3,300 places for Nordland and 450 for OxfordRobot car we only evaluated 2,700 and 360 places, respectively, ignoring the first 20% (see [Sect.4B Datasets](https://ieeexplore.ieee.org/document/10610918)). This can be modified with the `--skip` argument, which is set to 4799 by default for the pretrained Nordland models.

### Nordland
VPRTempo was developed and tested using the [Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset) traversal dataset. This software will work for either the full-resolution or down-sampled datasets, however our paper details the full-resolution datasets.

To simplify first usage, we have set the defaults in `VPRTempo.py` to train and test on a small subset of Nordland data. We recommend [downloading Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset) and using the `./src/nordland.py` script to unzip and organize the images into the correct file and naming structure.

For convenience, all data should be organised in the `./dataset` folder in the following way in order to train the network on multiple traversals of the same location.

```
--dataset
  |--summer
  |--spring
  |--fall
  |--winter
```

### Oxford RobotCar
In order to train and test on Oxford RobotCar, you will need to [register an account](https://mrgdatashare.robots.ox.ac.uk/register/) to get access to download the dataset before proceeding. We use 3 traverses (sun 2015-08-12-15-04-18, dusk 2014-11-21-16-07-03, and rain 2015-10-29-12-18-17) recorded from the `stero_left` camera, which can be downloaded using the [RobotCarDataset-Scraper](https://github.com/mttgdd/RobotCarDataset-Scraper) in the following way:

```console
# Copy the orc_lists.txt from this repo into the RobotCarDataset-Scraper repo
python scrape_mrgdatashare.py --choice_sensors stereo_left --choice_runs_file orc_list.txt --downloads_dir ~/VPRTempo/vprtempo/dataset/orc --datasets_file datasets.csv --username USERNAME --password PASSWORD
```

Next, use our helper script `process_orc.py` to demosaic and denoise the downloaded images. You'll need to download the [robotcar-dataset-sdk](https://github.com/ori-mrg/robotcar-dataset-sdk) repository and place the `process_orc.py` file into the `python` directory of the repository. Modify the `base_path` variable of `process_orc.py` to the location of your downloaded images.

```console
# Navigate to python directory, ensure process_orc.py and orc.csv are in this directory
cd ~/robotcar-dataset-sdk/python

# Run the demosaic and denoise
python process_orc.py
```

### Custom Datasets
To define your own custom dataset to use with VPRTempo, you will need to follow the conventions for [PyTorch Datasets & Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). We have included a convenient script `./vprtempo/src/create_data_csv.py` which will generate a .csv file that can be used to load custom datasets for training and inferencing. Simply modify the `dataset_name` variable to the folder containing your images.

To train a new model with a custom dataset, you can do the following.

```console
# Train new model - requires .csv file generated by create_data_csv.py
python main.py --train_new_model --dataset <your custom database name> --database_dirs <your custom database name>

# Test new model
python main.py --database_dirs <your custom database name> --dataset <your custom query name> --query_dir <your custom query name>
```
If image names are equivelant between database and query directories, you can simply use the one .csv file for both as in the example of Nordland and Oxford RobotCar.

## Usage
Running VPRTempo and VPRTempoQuant is handlded by `main.py`, which can be operated either through the command terminal or directly running the script. See below for more details.
### Prerequisites
* Training and testing data is organized as above (see **Datasets** on how to set up the Nordland dataset)
* The VPRTempo dependencies have been installed and/or the conda environment has been activated

### Pretrained models
We provide two pretrained models, for `VPRTempo` and `VPRTempoQuant`, that have learned a 500 place sequence from two Nordland traversals (Spring & Fall) which can be used to inference with Summer or Winter. To get the pretrained models, please download them [here](https://www.dropbox.com/scl/fi/ysfz7t7ek6h0pslwq9hd4/VPRTempo_pretrained_models.zip?rlkey=thg0rhn0hjsyov6zov63ni11o&st=nvimet71&dl=0).

### Run the inference network
The `main.py` script handles running the inference network, there are two options:

#### Command terminal
```console
python main.py
```
<p style="width: 100%; display: block; margin-left: auto; margin-right: auto">
  <img src="./assets/main_example.gif" alt="Example of the base VPRTempo networking running"/>
</p>

To run the quantized network, parse the `--quantize` argument.
```console
python main.py --quantize
```
<p style="width: 100%; display: block; margin-left: auto; margin-right: auto">
  <img src="./assets/mainquant_example.gif" alt="Example of the quantized VPRTempo networking running"/>
</p>


### Train new network
If you do not wish to use the pretrained models or you would like to train your own, we can parse the `--train_new_model` flag to `main.py`. Note, if a pretrained model already exists you will be prompted if you would like to retrain it.
```console
# For VPRTempo
python main.py --train_new_model

# For VPRTempoQuant
python main.py --train_new_model --quantize
```
<p style="width: 100%; display: block; margin-left: auto; margin-right: auto">
  <img src="./assets/train_example.gif" alt="Example of the training VPRTempo networking running"/>
</p>

Similarly above, if you wish to run the training through an IDE then change the `bool` flag for `train_new_model` to `True`.

## Tutorials
We provide a series of Jupyter Notebook [tutorials](https://github.com/AdamDHines/VPRTempo-quant/tree/main/tutorials) that go through the basic operations and logic for VPRTempo and VPRTempoQuant. 

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/QVPR/VPRTempo/issues).
