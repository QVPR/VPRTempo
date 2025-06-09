# VPRTempo - A Temporally Encoded Spiking Neural Network for Visual Place Recognition
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)
[![stars](https://img.shields.io/github/stars/QVPR/VPRTempo.svg?style=flat-square)](https://github.com/QVPR/VPRTempo/stargazers)
[![Downloads](https://static.pepy.tech/badge/vprtempo)](https://pepy.tech/project/vprtempo)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/vprtempo.svg)](https://anaconda.org/conda-forge/vprtempo)
[![GitHub repo size](https://img.shields.io/github/repo-size/QVPR/VPRTempo.svg?style=flat-square)](./README.md)


This repository contains code for [VPRTempo](https://vprtempo.github.io), a spiking neural network that uses temporally encoding to perform visual place recognition tasks. The network is based off of [BLiTNet](https://arxiv.org/pdf/2208.01204.pdf) and adapted to the [VPRSNN](https://github.com/QVPR/VPRSNN) framework. 

<p style="width: 50%; display: block; margin-left: auto; margin-right: auto">
  <img src="./assets/vprtempo_example.gif" alt="VPRTempo method diagram"/>
</p>

VPRTempo is built on a [torch.nn](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) framework and employs custom learning rules based on the temporal codes of spikes in order to train layer weights. 

In this repository, we provide two networks: 
  - `VPRTempo`: Our base network architecture to perform visual place recognition (fp32)
  - `VPRTempoQuant`: A modified base network with [Quantization Aware Training (QAT)](https://pytorch.org/docs/stable/quantization.html) enabled (int8)

To use VPRTempo, please follow the instructions below for installation and usage.

## :star: Update v1.1.10: What's new?
  - Adding [pixi](https://prefix.dev/) for easier setup and reproducibility :rocket:
  - Fixed repo size issue for a more compact download :chart_with_downwards_trend:

## Quick start
For simplicity and reproducibility, VPRTempo uses [pixi](https://prefix.dev/) to install and manage dependencies. If you do not already have pixi installed, run the following in your command terminal:

```console
curl -fsSL https://pixi.sh/install.sh | bash
```
For more information, please refer to the [pixi documentation](https://prefix.dev/docs/prefix/overview).

### Get the repository
Get the latest VPRTempo code and navigate to the project directory by running the following in your command terminal:
```console
git clone https://github.com/QVPR/VPRTempo.git
cd VPRTempo
```

### Run the demo
To quickly evaluate VPRTempo, we provide a pre-trained network trained on 500 places from the [Nordland](https://nrkbeta.no/2013/01/15/nordlandsbanen-minute-by-minute-season-by-season/) dataset. Run the following in your command terminal to run the demo:
```console
pixi run demo
```
_Note: this will start a download of the models and datasets (~600MB), please ensure you have enough disk space before proceeding._

### Train and evaluate a new model
Training and evaluating a new model is quick and easy, simply run the following in your command terminal to re-train and evaluate the demo model:

```console
pixi run train
pixi run eval
```
_Note: You will be prompted if you want to retrain the pre-existing network._

### Use the quantized models
For training and evaluation of the 8-bit quantized model, run the following in your command terminal:

```console
pixi run train_quant
pixi run eval_quant
```

### Alternative dependency install
Dependencies for VPRTempo can alternatively be installed in a conda environment. We recommend [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) and run the following in your command terminal:

```console
micromamba create -n vprtempo -c conda-forge vprtempo
micromamba activate vprtempo
```
_Note: Whilst we do have a PyPi package, we do not recommend using pip to install dependencies for VPRTempo._

## Datasets
VPRTempo was developed to be simple to train and test a variety of datasets. Please see the information below about recreating our results for the Nordland and Oxford RobotCar datasets and setting up custom datasets.

### Nordland
VPRTempo was developed and tested using the [Nordland](https://nrkbeta.no/2013/01/15/nordlandsbanen-minute-by-minute-season-by-season/) dataset. To download the full dataset, please visit [this repository](https://huggingface.co/datasets/Somayeh-h/Nordland?row=0). Once downloaded, place dataset folders into the VPRTempo directory as follows:

```
|__./vprtempo
    |___dataset
        |__summer
        |__spring
        |__fall
        |__winter
```

To replicate the results in our paper, run the following in your command terminal:

```console
pixi run nordland_train
pixi run nordland_eval
```

Alternatively, specify the data directory using the following argument:

```console
pixi run nordland_train --data_dir <YOUR_DIRECTORY>
pixi run nordland_eval --data_dir <YOUR_DIRECTORY>
```

### Oxford RobotCar
In order to train and test on Oxford RobotCar, you will need to [register an account](https://mrgdatashare.robots.ox.ac.uk/register/) to get access to download the dataset and process the images before proceeding. For more information, please refer to the [documentation]().

Once fully processed, to replicate the results in our paper run the following in your command terminal:

```console
pixi run orc_train
pixi run orc_eval
```

### Custom Datasets
To define your own custom dataset to use with VPRTempo, simply follow the same dataset structure defined above for Nordland. A `.csv` file of the image names will be required for the dataloader. 

We have included a convenient script `./vprtempo/src/create_data_csv.py` which will generate the necessary file. Simply modify the `dataset_name` variable to the folder containing your images.

To train a new model with a custom dataset, you can do the following.

```console
pixi run train --dataset <your custom database name> --database_dirs <your custom database name>
pixi run eval --database_dirs <your custom database name> --dataset <your custom query name> --query_dir <your custom query name>
```

## License & Citation
This repository is licensed under the [MIT License](./LICENSE). If you use our code, please cite our IEEE ICRA [paper](https://ieeexplore.ieee.org/document/10610918):
```
@inproceedings{hines2024vprtempo,
      title={VPRTempo: A Fast Temporally Encoded Spiking Neural Network for Visual Place Recognition}, 
      author={Adam D. Hines and Peter G. Stratton and Michael Milford and Tobias Fischer},
      year={2024},
      pages={10200-10207},
      booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}     
}
```

## Tutorials
We provide a series of Jupyter Notebook [tutorials](https://github.com/QVPR/VPRTempo/tree/main/tutorials) that go through the basic operations and logic for VPRTempo and VPRTempoQuant. 

## Issues, bugs, and feature requests
If you encounter problems whilst running the code or if you have a suggestion for a feature or improvement, please report it as an [issue](https://github.com/QVPR/VPRTempo/issues).
