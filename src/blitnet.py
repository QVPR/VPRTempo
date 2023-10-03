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
import numpy as np
import pdb
import torch
import gc

import torch.nn as nn

from timeit import default_timer

##################################
# Add a set of random connections between layers
#  W_range:    weight range [lo,hi]
#  p:          initial connection probability
#  stdp_rate:  STDP rate (0=no STDP)

def addWeights(W_range=[-1,0,1],p=[1,1],stdp_rate=0.001,dims=None,
               num_modules=1):

    # get torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    # Check constraints etc
    if np.isscalar(W_range): W_range = [W_range,W_range]
    
    # determine dimensions of the weight matrices
    nrow = dims[0]
    ncol = dims[1]
    
    # calculate mean and std for normal distributions
    inWmn = (W_range[0]+W_range[1])/2.0
    inWsd = (W_range[1]-W_range[0])/6.0
    exWmn = (W_range[1]+W_range[2])/2.0
    exWsd = (W_range[2]-W_range[1])/6.0
    
    # Initialize excW and inhW as empty tensors
    excW = torch.empty((0, nrow, ncol), device=device)
    inhW = torch.empty((0, nrow, ncol), device=device)
    
    # Loop through modules and add excitatory and inhibitory weights
    for n in range(num_modules):
        if n == 0:  # first weights to be appended
            # excitatory weights
            excW = torch.cat((excW, 
                              torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=exWmn, std=exWsd), 
                                              0)), 
                             0)
            # inhibitory weights
            inhW = torch.cat((inhW, 
                              torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=inWmn, std=inWsd), 
                                              0)), 
                             0)
        else:  # stack new weights onto appended weight
            excW = torch.cat((excW, 
                              torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=exWmn, std=exWsd), 
                                              0)), 
                             0)
            inhW = torch.cat((inhW, 
                              torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=inWmn, std=inWsd), 
                                              0)), 
                             0)
        
        # Remove negative excitatory weights
        excW[n][excW[n] < 0] = 0.0
        # Remove positive inhibitory weights
        inhW[n][inhW[n] > 0] = 0.0

        # remove connections based on exc and inh probabilities
        setzeroExc = np.random.rand(nrow,ncol) > p[0]
        setzeroInh = np.random.rand(nrow,ncol) > p[1]
        
        # add current
        if n == 0:
            I = torch.zeros(nrow, device=device)
            I = torch.unsqueeze(I,0)
        else:
            I = torch.concat((I,torch.unsqueeze(torch.zeros(nrow, device=device),0)),0)
        
        # remove connections based on calculated indexes
        if setzeroExc.any():
           excW[n,:,:][setzeroExc] = 0.0 # excitatory connections
        if setzeroInh.any():
            inhW[n,:,:][setzeroInh] = 0.0 # inhibitory connections
    
        # Normalise the weights (except fast inhib weights)
        nrmExc = torch.linalg.norm(excW[len(excW)-1],ord=1,axis=0)
        nrmInh = torch.linalg.norm(inhW[len(inhW)-1],ord=1,axis=0)
        nrmExc[nrmExc==0.0] = 1.0
        nrmInh[nrmInh==0.0] = 1.0
        excW[n] = excW[n,:,:]/nrmExc
        inhW[n] = inhW[n,:,:]/nrmInh
    
    havconnExc = excW > 0
    havconnInh = inhW < 0
        
    return excW, inhW, I, havconnExc, havconnInh
    
##################################
# Normalise all the firing rates
#  net: BITnet instance

def norm_rates(pre_layer,post_layer):
    # put layers into a list
    layers = [pre_layer,post_layer]
    
    for layer in layers:
        if layer.have_rate and layer.eta_ip > 0.0:
           # update = torch.add(layer.thr,
            #    torch.mul(layer.eta_ip, torch.sub(layer.x, layer.fire_rate))) 
            # Replace the original layer.thr with the updated one
            layer.thr = nn.Parameter(torch.where(layer.thr + layer.eta_ip * (layer.x - layer.fire_rate) < 0, 
                                                 torch.zeros_like(layer.thr), 
                                                 layer.thr + (layer.eta_ip * (layer.x - layer.fire_rate))))

            
    torch.cuda.empty_cache()
            
##################################
# Normalise inhib weights to balance input currents
#  net: BITnet instance

def norm_inhib(layer):
    if torch.any(layer.inhW):
        if layer.eta_ip != 0:
            updated_inhW = layer.inhW + torch.mul(torch.mul(layer.x_input, layer.inhW), 
                                                  layer.eta_stdp*50)
            layer.inhW = nn.Parameter(torch.where(updated_inhW > 0.0, 
                                                  torch.tensor(-0.000001, device=updated_inhW.device), 
                                                  updated_inhW))

            
def const_thr(layer_pre, layer_post):
    layers = [layer_pre, layer_post]
    for layer in layers:
        if torch.any(layer.const_inp):
            layer.x_input.fill_(0.0)  
            layer.x_input.add_(layer.const_inp)
        
        layer.x.detach().add_(torch.clamp(torch.sub(layer.x_input, layer.thr),
                                          0.0, 0.9))



def calc_spikes(post_layer, spikes):  

    post_layer.x_input.detach().add_(torch.bmm(spikes, post_layer.excW))
    post_layer.x_input.detach().add_(torch.bmm(spikes, post_layer.inhW))
    
    if post_layer.spk_force: 
        post_layer.x_calc.detach().add_(torch.clamp(torch.sub(post_layer.x_input, post_layer.thr),
                                                    min=0.0, max=0.9))
    else: 
        post_layer.x = torch.full_like(post_layer.x,0)
        post_layer.x.detach().add_(torch.clamp(torch.sub(post_layer.x_input, post_layer.thr), 
                                                min=0.0, max=0.9))
      

##################################
# Calculate STDP
#  net: BITnet instance

def calc_stdp(pre_layer,post_layer,spikes,idx=0):
    layers = [pre_layer,post_layer]
    
    # Spike Forcing has special rules to make calculated and forced spikes match
    if layers[-1].spk_force: # will run for the output layer
        shape = [len(layers[-1].excW[:, 0, 0]),
             len(layers[-1].excW[0, :, 0]),
             len(layers[-1].excW[0, 0, :])]
        # Get the output neuron index
        idx_sel = torch.arange(int(idx[0]),int(idx[0])+1,device=layers[-1].device,dtype=int)   
    
        # Difference between forced and calculated spikes
        layers[-1].x = torch.full_like(layers[-1].x,0)
        xdiff = torch.clamp(layers[-1].x.index_fill_(-1,idx_sel,0.5) - layers[-1].x_calc,
                            min=0,max=1)

        # Threshold rules - lower it if calced spike is smaller (and vice versa)
        layers[-1].thr = nn.Parameter(layers[-1].thr - 
                                      torch.sign(xdiff)*torch.abs(layers[-1].eta_stdp)/10)
        layers[-1].thr = nn.Parameter(layers[-1].thr - 
                                      torch.sign(xdiff)*torch.abs((layers[-1].eta_stdp*-1))/10)
        layers[-1].thr = nn.Parameter(layers[-1].thr.clamp(min=0, max=1))

        # Pre and Post spikes tiled across and down for all synapses
        if layers[0].have_rate:
            # Modulate learning rate by firing rate (low firing rate = high learning rate)
            mpre = spikes/layers[0].fire_rate
        else:
            mpre = spikes
        pre = torch.tile(torch.reshape(mpre,(shape[0],shape[1],1)),(1,shape[2]))
        post = torch.tile(xdiff,(shape[1],1))

        # Apply the weight changes
        layers[-1].excW = nn.Parameter(layers[-1].excW + 
                                       (pre*post*layers[-1].havconnExc)*layers[-1].eta_stdp)
        layers[-1].inhW = nn.Parameter(layers[-1].inhW + 
                                       (-pre*post*layers[-1].havconnInh)*(layers[-1].eta_stdp*-1))
    
    # Normal STDP
    else:
    
        # Assuming layers is a predefined list containing your layer objects
        
        shape = [len(layers[-1].excW[:, 0, 0]),
                 len(layers[-1].excW[0, :, 0]),
                 len(layers[-1].excW[0, 0, :])]
        
        # Tile out pre- and post-spikes
        pre = torch.tile(torch.reshape(spikes, (shape[0], shape[1], 1)), (1, shape[2]))
        post = torch.tile(layers[-1].x, (shape[1], 1))

        # Apply the weight changes
        layers[-1].excW = nn.Parameter(layers[-1].excW + ((0.5 - post) * (pre > 0) * (post > 0) * layers[-1].havconnExc) * layers[-1].eta_stdp)
        layers[-1].inhW = nn.Parameter(layers[-1].inhW + ((0.5 - post) * (pre > 0) * (post > 0) * layers[-1].havconnInh) * (layers[-1].eta_stdp * -1))
        
    # In-place clamp for excitatory and inhibitory weights
    # Apply clamping only where the mask is True (non-zero elements)
    layers[-1].excW = nn.Parameter(torch.where(layers[-1].havconnExc,
                                  layers[-1].excW.clamp(min=0.000001, max=10.0),
                                  layers[-1].excW))
    
    # Apply clamping only where the mask is True (non-zero elements)
    layers[-1].inhW = nn.Parameter(torch.where(layers[-1].havconnInh,
                                  layers[-1].inhW.clamp(min=-10.0, max=-0.000001),
                                  layers[-1].inhW))
    
    torch.cuda.empty_cache()

##################################
# Run the simulation
#  net: BITnet instance
#  n_steps: number of steps

def runSim(pre_layer,post_layer,spikes,idx):

    # Propagate spikes from pre to post neurons
    const_thr(pre_layer,post_layer)
    calc_spikes(post_layer,spikes)

    # Calculate STDP weight changes
    calc_stdp(pre_layer,post_layer,spikes,idx)

    # Normalise firing rates and inhibitory balance
    norm_rates(pre_layer,post_layer)
    norm_inhib(post_layer)

    torch.cuda.empty_cache()
    
    del spikes

def testSim(layers,layer_num,spikes,idx):
    
    # run the test system through all specified layers to get an output
    for n in range(layer_num):
        calc_spikes(layers['layer'+str(n+1)],spikes)
        
    return layers['layer'+str(n+1)].x