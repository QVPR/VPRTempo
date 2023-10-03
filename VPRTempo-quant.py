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

import os
import torch
import gc
#gc.disable()
import sys
sys.path.append('./src')
sys.path.append('./weights')
sys.path.append('./settings')
sys.path.append('./output')
sys.path.append('./dataset')
sys.path.append('./config')

import blitnet as bn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from config import configure
from dataset import CustomImageDataset, SetImageAsSpikes, ProcessImage
from torch.utils.data import DataLoader
from timeit import default_timer
from tqdm import tqdm


class SNNLayer(nn.Module):
    def __init__(self, previous_layer=None,dims=[0,0,0],thr_range=[0,0], 
                 fire_rate=[0,0],ip_rate=0,stdp_rate=0,const_inp=[0,0],p=[1,1],
                 assign_weight=False,spk_force=False):
        super(SNNLayer, self).__init__()
        configure(self)
        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Check constraints etc
        if np.isscalar(thr_range): thr_range = [thr_range, thr_range]
        if np.isscalar(fire_rate): fire_rate = [fire_rate, fire_rate]
        if np.isscalar(const_inp): const_inp = [const_inp, const_inp]
        
        # Initialize Tensors
        self.dim = torch.tensor(dims, dtype=torch.int)
        self.x = torch.zeros(dims, device=self.device,requires_grad=False)
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
        self.p = p
        self.dims = dims
        
        # Additional State Variables
        self.set_spks = []
        self.sspk_idx = 0
        self.spikes = torch.empty([], dtype=torch.float64)
        self.spk_force = spk_force
        
        # Weights (if applicable)
        if assign_weight:
            self.excW, self.inhW, self.I, self.havconnExc, self.havconnInh = bn.addWeights(p=self.p,
                                                 stdp_rate=self.eta_stdp,
                                                 dims=[previous_layer.dims[2],
                                                       dims[2]],
                                                 num_modules=self.number_modules)
        
class SNNTrainer(nn.Module):
    def __init__(self):
        super(SNNTrainer, self).__init__()
        configure(self)
        
        # Set the device and model on defined device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.to(self.device)
        
        # Set up the input layer
        self.input_layer = SNNLayer(dims=[self.number_modules,1,self.input])
        
        # Set up the feature layer
        self.feature_layer = SNNLayer(previous_layer=self.input_layer,
                                      dims=[self.number_modules,1,self.feature],
                                      thr_range=[0,0.5],
                                      fire_rate=[0.2,0.9],
                                      ip_rate=0.15,
                                      stdp_rate=0.005,
                                      const_inp=[0,0.1],
                                      p=[0.1,0.5],
                                      assign_weight=True)
        
        # Set up the output layer
        self.output_layer = SNNLayer(previous_layer=self.feature_layer,
                                    dims=[self.number_modules,1,self.output],
                                    stdp_rate=0.005,
                                    assign_weight=True,
                                    spk_force=True)
        
        # Define number of layers (will run training on all layers)
        self.layers = {'layer0':self.input_layer,
                       'layer1':self.feature_layer,
                       'layer2':self.output_layer}
    
    def train_model(self, train_loader):
        
        # If using CUDA, run a dummy torch.bmm to 'spool-up' operations
        if self.device.type == "cuda":
            # Create some dummy tensors on CUDA
            dummy_a = torch.randn(10, 10, device=self.device)
            dummy_b = torch.randn(10, 10, device=self.device)
            
            # Perform a dummy bmm operation
            torch.bmm(dummy_a.unsqueeze(0), dummy_b.unsqueeze(0))
        
        # Run the training for each layer, using defined parameters
        for layer in range(len(self.layers)-1):
            
            # Define in and out layer
            in_layer = self.layers['layer'+str(layer)]
            out_layer = self.layers['layer'+str(layer+1)]
            
            # Output the learning rates for annealment during training
            n_initstdp = out_layer.eta_stdp.detach() # STDP
            n_initip = out_layer.eta_ip.detach() # ITP
            
            # Run the training for the input to feature layer for specified epochs
            for n in tqdm(range(self.epoch)):
                mod = 0 # Used to determine the learning rate annealment
                for images, labels in train_loader: 
                    # Put input images on device (CPU, CUDA)
                    images = images.to(self.device)
                    
                    # Set spikes from input images
                    make_spikes = SetImageAsSpikes(self.intensity)
                    spikes = make_spikes(images)
                    
                    # Put labels on device (CPU, CUDA)
                    labels = labels.to(self.device)
                    idx = labels/self.filter

                    # Layers don't include loaded input, calculate network spikes for up to training layers
                    if layer != 0:
                        spikes = bn.testSim(self.layers,layer,spikes,idx)
                        
                    # Run one timestep of the training input to feature layer
                    bn.runSim(in_layer, out_layer, spikes, idx)
                    
                    # Anneal the learning rate
                    if np.mod(mod,10)==0:
                        pt = pow(float(self.T-mod)/self.T,self.annl_pow)
                        out_layer.eta_ip = torch.mul(n_initip,pt)
                        out_layer.eta_stdp = torch.mul(n_initstdp,pt)
                    mod += 1
            
            # Once layers have finished training, turn off any learning
            out_layer.eta_ip = 0
            out_layer.eta_stdp = 0
            out_layer.excW.requires_grad_(False)
            out_layer.inhW.requires_grad_(False)
            out_layer.thr.requires_grad_(False)
        
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved at {self.model_path}")

class SNNModel(nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()
        # Configure the network
        configure(self) 
        self.model = self.to(self.device)
        
    def forward(self, x):
        # Define the forward pass to transform the input x to an output
        # TODO: Replace with actual forward pass code
        out = bn.testSim(self)  # Assuming testSim is a function that can perform the forward pass
        return out

if __name__ == "__main__":
    # initialize the model and image transforms
    model = SNNModel()
    image_transform = ProcessImage(model.dims,model.patches)
    
    # TODO: check for existence of pre-existing model, if not then run training
    # just run training and testing in tandem for now
    train_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                      img_dirs=model.training_dirs,
                                      transform=image_transform,
                                      skip=model.filter,
                                      max_samples=model.number_training_images,
                                      modules=model.number_modules)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=model.number_modules, 
                              shuffle=False,
                              num_workers=4,
                              persistent_workers=True)
    
    # initialize the training model
    trainer = SNNTrainer()
    trainer.train_model(train_loader)
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during testing
        for inputs, targets in test_dataset:
            outputs = model(inputs)  # This calls the forward method and gets the model’s outputs
            # TODO: Compute your evaluation metric(s) by comparing outputs to targets
