#MIT License

#Copyright (c) 2023 Adam Hines, Peter G Stratton, Michael Milford, Tobias Fischer

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

'''
Imports
'''

import pickle
import os
import torch
import gc
import timeit

import logging
import sys
sys.path.append('./src')
sys.path.append('./weights')
sys.path.append('./settings')
sys.path.append('./output')

import blitnet as bn
import utils as ut
import validation as val
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from os import path
from datetime import datetime


class SNNLayer(nn.Module):
    def __init__(self, dims, thr_range, fire_rate, ip_rate, stdp_rate, const_inp,
                 assign_weight):
        super(SNNLayer, self).__init__()

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Check constraints etc
        if np.isscalar(thr_range): thr_range = [thr_range, thr_range]
        if np.isscalar(fire_rate): fire_rate = [fire_rate, fire_rate]
        if np.isscalar(const_inp): const_inp = [const_inp, const_inp]
        
        # Initialize Tensors
        self.dim = torch.tensor(dims, dtype=torch.int)
        self.x = torch.zeros(dims, device=self.device)
        self.x_prev = torch.zeros(dims, device=self.device)
        self.x_calc = torch.zeros(dims, device=self.device)
        self.x_input = torch.zeros(dims, device=self.device)
        self.x_fastinp = torch.zeros(dims, device=self.device)
        self.eta_ip = torch.tensor(ip_rate, device=self.device)
        self.eta_stdp = torch.tensor(stdp_rate, device=self.device)
        
        # Initialize Parameters
        self.thr = nn.Parameter(torch.zeros(dims, device=self.device).uniform_(thr_range[0], thr_range[1]))
        self.fire_rate = torch.zeros(dims, device=self.device).uniform_(fire_rate[0], fire_rate[1])
        self.have_rate = torch.any(self.fire_rate[:,:,0] > 0.0).to(self.device)
        self.const_inp = torch.zeros(dims, device=self.device).uniform_(const_inp[0], const_inp[1])
        
        # Additional State Variables
        self.set_spks = []
        self.sspk_idx = 0
        self.spikes = torch.empty([], dtype=torch.float64)
        
        # Store the layer numbers
        self.layers.append(len(self.layers))
        
        # Weights (if applicable)
        if assign_weight:
            self.excW, self.inhW = bn.addWeights(self.layers)
        
class SNNModel(nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()
        # define layer parameters
        self.number_modules = 1 # set the numnber of expert modules
        self.module_max = 100 # set the maximum number of places per module
        self.imWidth = 28 # set the pixel width (after pre-processing)
        self.imHeight = 28 # set the pixel height (after pre-processing)
        self.dim = int(self.imWidth*self.imHeight) # calculate the input layer size
        self.layers = []
        
        # initialize new net 
        self.net = bn.newNet(self.number_modules,self.dim)
        
        # add the layers
        # input layer
        self.input_layer = SNNLayer(
            [self.number_modules,1,self.dim],
            0,0,0,0,0,False)
        
        # feature layer
        self.feature_layer = SNNLayer(
            [self.number_modules,1,int(self.dim*2)],
            [0,0.5],
            [0.2,0.9],
            0.15,
            0.005,
            [0,0.1],
            True)
        
        # output layer
        self.output_layer = SNNLayer(
            [self.number_modules,1,self.module_max],
            0,0,0,0,[0,0],True) # output layer
        
        
    def forward(self):
        # run the network to get the output here
        bn.testSim()
        
def configure_model(model):
    model.dataset = 'nordland'
    model.trainingPath = '/home/adam/data/nordland/'
    model.testPath = '/home/adam/data/nordland/'
    model.number_modules = 1
    model.number_training_images = 100
    model.number_testing_images = 100
    model.locations = ["spring", "fall"]
    model.test_location = "summer"
    model.filter = 8
    model.validation = True
    model.log = False

    assert (len(model.dataset) != 0), "Dataset not defined, see README.md for details on setting up images"
    assert (os.path.isdir(model.trainingPath)), "Training path not set or path does not exist, specify for model.trainingPath"
    assert (os.path.isdir(model.testPath)), "Test path not set or path does not exist, specify for model.testPath"
    assert (os.path.isdir(model.trainingPath + model.locations[0])), "Images must be organized into folders based on locations, see README.md for details"
    assert (os.path.isdir(model.testPath + model.test_location)), "Images must be organized into folders based on locations, see README.md for details"
    
    '''
    NETWORK SETTINGS
    '''
    model.num_patches = 7
    model.intensity = 255
    model.location_repeat = len(model.locations)
    
    model.epoch = 4
    model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model.device.type == "cuda":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.cuda.set_device(model.device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.init()
        torch.cuda.synchronize(device=model.device)
    model.T = int((model.number_training_images / model.number_modules) * model.location_repeat)
    model.annl_pow = 2
    model.imgs = {'training': [], 'testing': []}
    model.ids = {'training': [], 'testing': []}
    model.spike_rates = {'training': [], 'testing': []}
    
    model.n_itp = 0.15
    
    model.test_true = False

    '''
    DATA SETTINGS
    '''
    with open('./' + model.dataset + '_imageNames.txt') as file:
        model.imageNames = [line.rstrip() for line in file]
        
    model.filteredNames = []
    for n in range(0, len(model.imageNames), model.filter):
        model.filteredNames.append(model.imageNames[n])
    del model.filteredNames[model.number_training_images:len(model.filteredNames)]
    
    model.fullTrainPaths = []
    for n in model.locations:
        model.fullTrainPaths.append(model.trainingPath + n + '/')
    
    now = datetime.now()
    model.output_folder = './output/' + now.strftime("%d%m%y-%H-%M-%S")
    os.mkdir(model.output_folder)
    
    model.logger = logging.getLogger("VPRTempo")
    model.logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename=model.output_folder + "/logfile.log",
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if model.log:
        model.logger.addHandler(logging.StreamHandler())
    
    model.logger.info('////////////')
    model.logger.info('VPRTempo - Temporally Encoded Visual Place Recognition v1.1.0-alpha')
    model.logger.info('Queensland University of Technology, Centre for Robotics')
    model.logger.info('')
    model.logger.info('© 2023 Adam D Hines, Peter G Stratton, Michael Milford, Tobias Fischer')
    model.logger.info('MIT license - https://github.com/QVPR/VPRTempo')
    model.logger.info('\\\\\\\\\\\\\\\\\\\\\\\\')
    model.logger.info('')
    model.logger.info('CUDA available: ' + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        model.logger.info('Current device is: ' + str(torch.cuda.get_device_name(current_device)))
    else:
        model.logger.info('Current device is: CPU')
    model.logger.info('')
    model.logger.info("~~ Hyperparameters ~~")
    model.logger.info('')
    model.logger.info('Firing threshold max: ' + str(model.theta_max))
    model.logger.info('Initial STDP learning rate: ' + str(model.n_init))
    model.logger.info('Intrinsic threshold plasticity learning rate: ' + str(model.n_itp))
    model.logger.info('Firing rate range: [' + str(model.f_rate[0]) + ', ' + str(model.f_rate[1]) + ']')
    model.logger.info('Excitatory connection probability: ' + str(model.p_exc))
    model.logger.info('Inhibitory connection probability: ' + str(model.p_inh))
    model.logger.info('Constant input: ' + str(model.c))
    model.logger.info('')
    model.logger.info("~~ Training and testing conditions ~~")
    model.logger.info('')
    model.logger.info('Number of training images: ' + str(model.number_training_images))
    model.logger.info('Number of testing images: ' + str(model.number_testing_images))
    model.logger.info('Number of training epochs: ' + str(model.epoch))
    model.logger.info('Number of modules: ' + str(model.number_modules))
    model.logger.info('Dataset used: ' + str(model.dataset))
    model.logger.info('Training locations: ' + str(model.locations))
    model.logger.info('Testing location: ' + str(model.test_location))

    model.training_out = './weights/' + str(model.input_layer) + 'i' + str(model.feature_layer) + 'f' + str(model.output_layer) + 'o' + str(model.epoch) + '/'

      
if __name__ == "__main__":
    
    # Instantiate model
    model = SNNModel()
    configure_model(model)
    
    model.forward()
    