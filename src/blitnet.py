#MIT License

#Copyright (c) 2023 Adam Hines, Peter Stratton, Michael Milford, Tobias Fischer

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
import torch

import torch.nn as nn
import numpy as np

from config import configure


class SNNLayer(nn.Module):
    def __init__(self, dims=[0,0],thr_range=[0,0],fire_rate=[0,0],ip_rate=0,
                 stdp_rate=0,const_inp=[0,0],p=[1,1],spk_force=False):
        super(SNNLayer, self).__init__()
        # Configure the network
        configure(self) # Sets the testing configuration
        # Device
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        # Check constraints etc
        if np.isscalar(thr_range): thr_range = [thr_range, thr_range]
        if np.isscalar(fire_rate): fire_rate = [fire_rate, fire_rate]
        if np.isscalar(const_inp): const_inp = [const_inp, const_inp]
        
        # Initialize Tensors
        self.x = torch.zeros([1, dims[-1]], device=self.device)
        self.x_prev = torch.zeros([1, dims[-1]], device=self.device)
        self.x_calc = torch.zeros([1, dims[-1]], device=self.device)
        self.x_input = torch.zeros([1, dims[-1]], device=self.device)
        self.x_fastinp = torch.zeros([1, dims[-1]], device=self.device)
        
        self.eta_ip = torch.tensor(ip_rate, device=self.device)
        self.eta_stdp = torch.tensor(stdp_rate, device=self.device)
        
        # Initialize Parameters
        self.thr = nn.Parameter(torch.zeros([1, dims[-1]], 
                                            device=self.device).uniform_(thr_range[0], 
                                                                         thr_range[1]))
        self.fire_rate = torch.zeros([1,dims[-1]], device=self.device).uniform_(fire_rate[0], fire_rate[1])
        
        # Sequentially set the feature firing rates (if any)
        if not torch.all(self.fire_rate==0).item():
            fstep = (fire_rate[1]-fire_rate[0])/dims[-1]
            
            for i in range(dims[-1]):
               self.fire_rate[:,i] = fire_rate[0]+fstep*(i+1)
                   
        self.have_rate = torch.any(self.fire_rate[:,0] > 0.0).to(self.device)
        self.const_inp = torch.zeros([1, dims[-1]], device=self.device).uniform_(const_inp[0], const_inp[1])
        self.p = p
        self.dims = dims
        
        # Additional State Variables
        self.set_spks = []
        self.sspk_idx = 0
        self.spikes = torch.empty([], dtype=torch.float64)
        self.spk_force = spk_force
    
        # Create the excitatory weights
        self.exc = nn.Linear(dims[0], dims[1], bias=False)
        self.exc.weight = self.addWeights(dims=dims,
                                             W_range=[0,1], 
                                             p=p[0])
        
        # Create the inhibitory weights
        self.inh = nn.Linear(dims[0], dims[1], bias=False)
        self.inh.weight = self.addWeights(dims=dims,
                                             W_range=[-1,0], 
                                             p=p[-1])
        
        # Output boolean reference of which neurons have connection weights
        self.havconnExc = self.exc.weight > 0
        self.havconnInh = self.inh.weight < 0

    def addWeights(self,W_range=[0,0],p=[0,0],dims=[0,0]):

        # Get torch device
       #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        device = torch.device("cpu") 
    
        # Check constraints etc
        if np.isscalar(W_range): W_range = [W_range,W_range]
        
        # Determine dimensions of the weight matrices
        nrow = dims[1]
        ncol = dims[0]
        
        # Calculate mean and std for normal distributions
        Wmn = (W_range[0]+W_range[1])/2.0
        Wsd = (W_range[1]-W_range[0])/6.0
        
        # Initialize weights as empty tensors
        W = torch.empty((0, nrow, ncol), device=device)
        
        # Normally disribute random weights
        W = torch.empty(nrow, ncol, device=device).normal_(mean=Wmn, std=Wsd)
            
        # Remove inappropriate weights based on sign from W_range
        if W_range[-1] != 0:
            # For excitatory weights
            W[W < 0] = 0.0
        else:
            # For inhibitory weights
            W[W > 0] = 0.0
    
        # Remove weights based on connection probability
        setzero = np.random.rand(nrow,ncol) > p
        if setzero.any():
           W[setzero] = 0.0
    
        # Normalise the weights
        nrm = torch.linalg.norm(W[len(W)-1],ord=1,axis=0)
        nrm[nrm==0.0] = 1.0
        W = nn.Parameter(W/nrm)
    
        return W

class BLiTNET(nn.Module):
    def __init__(self,layer=None,spikes=None,idx=None,fr=None,testCount=None,
                 layer_lst=None):
        super(BLiTNET, self).__init__()
        
        # Define the layer & spikes to be parsed through BLiTNET
        self.layer = layer
        self.spikes = spikes
        
        # For spike forcing, define the output idx and pre-layer fire rate
        self.idx = idx
        self.fr = fr
        
        # For running testing, determine number of layers to iterate through for output
        self.testCount = testCount
        self.layer_lst = layer_lst
        
    def add_input(self):
        # Add the constant input
        self.layer.x_input += self.layer.const_inp
    
    def calc_spikes(self):  
        # Use nn.Linear to perform multiplication of spikes to weights
        self.layer.x_input += self.layer.exc(self.spikes)
        self.layer.x_input += self.layer.inh(self.spikes)
        
        # Clamp outputs between 0 and 0.9 after subtracting thresholds from input
        if self.layer.spk_force: 
            self.layer.x_calc = torch.clamp(torch.sub(self.layer.x_input, self.layer.thr), min=0.0, max=0.9)
        else: 
            self.layer.x = torch.clamp(torch.sub(self.layer.x_input, self.layer.thr), min=0.0, max=0.9)
    
    def calc_stdp(self):
        # Spike Forcing has special rules to make calculated and forced spikes match
        if self.layer.spk_force:
            
            # Get layer dimensions
            shape = self.layer.exc.weight.data.shape
            
            # Get the output neuron index
            idx_sel = torch.arange(int(self.idx[0]), int(self.idx[0]) + 1, 
                                   device=self.layer.device, 
                                   dtype=int)
    
            # Difference between forced and calculated spikes
            self.layer.x = torch.full_like(self.layer.x, 0)
            xdiff = torch.clamp(self.layer.x.index_fill_(-1, idx_sel, 0.5) - self.layer.x_calc, min=0, max=1)
    
            # Threshold rules - lower it if calced spike is smaller (and vice versa)
            self.layer.thr.data -= torch.sign(xdiff) * torch.abs(self.layer.eta_stdp) / 10
            self.layer.thr.data -= torch.sign(xdiff) * torch.abs((self.layer.eta_stdp * -1)) / 10
            self.layer.thr.data = self.layer.thr.data.clamp(min=0, max=1)
    
            # Pre and Post spikes tiled across and down for all synapses
            if self.fr == None:
                mpre = self.spikes
            else:
                # Modulate learning rate by firing rate (low firing rate = high learning rate)
                mpre = self.spikes/self.fr
                
            # Tile out pre- and post- spikes for STDP weight updates    
            pre = torch.tile(torch.reshape(mpre, (shape[1], 1)), (1, shape[0]))
            post = torch.tile(xdiff, (shape[1], 1))
    
            # Apply the weight changes
            self.layer.exc.weight.data += ((pre * post * self.layer.havconnExc.T) * 
                                           self.layer.eta_stdp).T
            self.layer.inh.weight.data += ((-pre * post * self.layer.havconnInh.T) * 
                                           (self.layer.eta_stdp * -1)).T
    
        # Normal STDP
        else:
            
            # Get layer dimensions
            shape = self.layer.exc.weight.data.shape
            
            # Tile out pre- and post-spikes
            pre = torch.tile(torch.reshape(self.spikes, (shape[1], 1)), (1, shape[0]))
            post = torch.tile(self.layer.x, (shape[1], 1))
            
            # Apply positive and negative weight changes
            self.layer.exc.weight.data += (((0.5 - post) * (pre > 0) * (post > 0) * 
                                      self.layer.havconnExc.T) * self.layer.eta_stdp).T
            self.layer.inh.weight.data += (((0.5 - post) * (pre > 0) * 
                                      (post > 0) * self.layer.havconnInh.T) * (self.layer.eta_stdp * -1)).T
    
        # In-place clamp for excitatory and inhibitory weights
        self.layer.exc.weight.data[self.layer.exc.weight.data < 0] = 1e-06
        self.layer.inh.weight.data[self.layer.inh.weight.data > 0] = -1e-06
        
        # Remove negative weights for excW and positive for inhW
        self.layer.exc.weight.data[self.layer.havconnExc] = self.layer.exc.weight.data[self.layer.havconnExc].clamp(min=1e-06, max=10)
        self.layer.inh.weight.data[self.layer.havconnInh] = self.layer.inh.weight.data[self.layer.havconnInh].clamp(min=-10, max=-1e-06)
     
        # Check if layer has target firing rate and an ITP learning rate
        if self.layer.have_rate and self.layer.eta_ip > 0.0:
            
            # Replace the original layer.thr with the updated one
            self.layer.thr.data += self.layer.eta_ip * (self.layer.x - self.layer.fire_rate)
            self.layer.thr.data[self.layer.thr.data < 0] = 0
        
        # Check if layer has inhibitory weights and an stdp learning rate
        if torch.any(self.layer.inh.weight.data).item() and self.layer.eta_stdp != 0:
            
            # Normalize the inhibitory weights using homeostasis
            inhW = self.layer.inh.weight.data.T
            self.layer.inh.weight.data += (torch.mul(self.layer.x_input,inhW) * self.layer.eta_stdp*50).T
            self.layer.inh.weight.data[self.layer.inh.weight.data > 0.0] = -1e-06       

            
    def runSim(self):
    
        # Propagate spikes from pre to post neurons
        self.add_input()
        self.calc_spikes()
    
        # Calculate STDP weight changes
        self.calc_stdp()


    def testSim(self):
        
        # run the test system through all specified layers to get an output
        for count, layer in enumerate(self.layer_lst):
            if count != self.testCount:
                self.layer = self.layer_lst[layer]
                self.layer.x.fill_(0.0)
                self.layer.x_input.fill_(0.0)
                self.calc_spikes()
                self.spikes = self.layer.x
            
        return self.layer.x