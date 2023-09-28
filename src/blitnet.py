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
            excW = torch.cat((excW, torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=exWmn, std=exWsd), 0)), 0)
            # inhibitory weights
            inhW = torch.cat((inhW, torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=inWmn, std=inWsd), 0)), 0)
        else:  # stack new weights onto appended weight
            excW = torch.cat((excW, torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=exWmn, std=exWsd), 0)), 0)
            inhW = torch.cat((inhW, torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=inWmn, std=inWsd), 0)), 0)
        
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
        
    return excW, inhW, I
    
##################################
# Normalise all the firing rates
#  net: BITnet instance

def norm_rates(pre_layer,post_layer):
    # put layers into a list
    layers = [pre_layer,post_layer]
    
    for layer in layers:
        if layer.have_rate and layer.eta_ip > 0.0:
            # adjust the firing threshold based on ideal firing rates and ITP
            update = torch.mul(layer.eta_ip, torch.sub(layer.x, layer.fire_rate)) 
            update[update<0.0] = 0.0
             
            # Replace the original layer.thr with the updated one
            layer.thr.detach().add_(update) 
    
    torch.cuda.empty_cache()
            
##################################
# Normalise inhib weights to balance input currents
#  net: BITnet instance

def norm_inhib(layer):
    if torch.any(layer.inhW):
        if layer.eta_ip != 0:
            layer.inhW.add_(torch.mul(torch.mul(layer.x_input, layer.inhW), -layer.eta_ip*50), alpha=1)
            layer.inhW[layer.inhW > 0.0] = -0.000001

                
##################################
# Propagate spikes thru the network
#  net: SORN instance
def add_spikesTest(net,device,dims):
    
    # Start with the constant input in the neurons of each layer
    for i,const in enumerate(net['const_inp']):

        net['x_input'][i] = torch.full_like(net['x_input'][i],0.0)
        net['x_input'][i] += net['const_inp'][i]
        # Find the threshold crossings (overwritten later if needed)
            
        net['x'][i] = torch.clamp((net['x_input'][i]-net['thr'][i]),0.0,0.9)
    
    # insert any predefined spikes
    start = dims * (net['step_num'] - 1)
    end = dims * net['step_num']
    for i in range(len(net['set_spks'])):
        if len(net['set_spks'][i]): # detect if any set_spks tensors
            start = dims * (net['step_num'] - 1)
            end = dims * net['step_num']
            index = torch.arange(start,end,device=device,dtype=int)
            if i == len(net['set_spks'])-1: # spike forcing
                net['x'][i] = net['set_spks'][i].index_fill_(-1,index,0.5)
            else:
                net['x'][i] = torch.index_select(net['set_spks'][i],-1,index)



def const_thr(layer_pre,layer_post):
    
    # put layers into a list
    layers = [layer_pre, layer_post]
    
    # Start with the constant input in the neurons of each layer
    for layer in layers:
        if torch.any(layer.const_inp):
            layer.x_input.fill_(0.0)  # In-place fill
            layer.x_input.add_(layer.const_inp)  # In-place addition
        
        # Find the threshold crossings (overwritten later if needed)
        layer.x = torch.clamp(torch.sub(layer.x_input, layer.thr), 0.0, 0.9)
        
    del layers


def calc_spikes(post_layer,spikes):  

    # Batch multiply the input spikes with the excitatory and inhibitory weights
    post_layer.x_input.add_(torch.bmm(spikes,post_layer.excW))
    post_layer.x_input.add_(torch.bmm(spikes,post_layer.inhW))

    # adjust the spikes based on threshold
    if post_layer.spk_force: # This layer has spike forcing, remember calculated spikes
        post_layer.x_calc = torch.clamp(torch.sub(post_layer.x_input,post_layer.thr),
                                    min=0.0,max=0.9)

    else: # Predefined spikes exist for this layer, remember the calculated ones
        post_layer.x = torch.clamp(torch.sub(post_layer.x_input,post_layer.thr),
                                        min=0.0,max=0.9)
    
    # update the x previous variable with calculated spikes
    post_layer.x_prev = post_layer.x.detach()            

##################################
# Calculate STDP
#  net: BITnet instance

def calc_stdp(pre_layer,post_layer,spikes):
    layers = [pre_layer,post_layer]
    #
    # Spike Forcing has special rules to make calculated and forced spikes match
    #
    if layers[-1].spk_force: # will run for the output layer

        # Diff between forced and calculated spikes
        xdiff = net['x'][layers[1]] - net['x_calc'][layers[1]]

        # Threshold rules - lower it if calced spike is smaller (and vice versa)
        net['thr'][layers[1]] -= torch.sign(xdiff)*torch.abs(torch.tensor(net['eta_stdp'][excW]))/10
        net['thr'][layers[1]] -= torch.sign(xdiff)*torch.abs(torch.tensor(net['eta_stdp'][inhW]))/10
        net['thr'][layers[1]][net['thr'][layers[1]]<0.0] = 0.0  # don't go -ve

        # Pre and Post spikes tiled across and down for all synapses
        if net['have_rate'][layers[0]]:
            # Modulate learning rate by firing rate (low firing rate = high learning rate)
            mpre = net['x'][layers[0]]/net['fire_rate'][layers[0]]
        else:
            mpre = net['x'][layers[0]]
        pre = torch.tile(torch.reshape(mpre,(shape[0],shape[1],1)),(1,shape[2]))
        post = torch.tile(xdiff,(shape[1],1))

        # Excitatory connections
        if net['eta_stdp'][excW] > 0:
            havconn = net['W'][excW]>0
            inc_stdp_exc = pre*post*havconn
        # Inhibitory connections
        if net['eta_stdp'][inhW] < 0:
            havconn = net['W'][inhW]<0
            inc_stdp_inh = -pre*post*havconn

        # Apply the weight changes
        net['W'][excW] += inc_stdp_exc*net['eta_stdp'][excW]
        net['W'][inhW] += inc_stdp_inh*net['eta_stdp'][inhW]
    
    #
    # Normal STDP
    #
    else:
        
        # Reshape spikes tensor and get post activation tensor
        pre = spikes.reshape(layers[-1].inhW.size(0), layers[-1].inhW.size(1), 1)
        post = layers[-1].x
        
        # Calculate the masks for excitatory and inhibitory connections
        havconnExc = layers[-1].excW > 0
        havconnInh = layers[-1].inhW < 0
                
        # Compute the increment for excW directly and apply the update
        layers[-1].excW.add_(((0.5 - post) * (pre > 0) * (post > 0) * havconnExc).mul_(layers[-1].eta_stdp))
        torch.cuda.empty_cache()
        # Compute the increment for inhW directly and apply the update
        layers[-1].inhW.add_(((0.5 - post) * (pre > 0) * (post > 0) * havconnInh).mul_(layers[-1].eta_stdp * -1))
        torch.cuda.empty_cache()
    
    # In-place clamp for excitatory and inhibitory weights
    layers[-1].excW = layers[-1].excW.clamp(min=0, max=10.0).detach()
    torch.cuda.empty_cache()
    layers[-1].inhW = layers[-1].inhW.clamp_(min=-10.0, max=0).detach()
    torch.cuda.empty_cache()


##################################
# Run the simulation
#  net: BITnet instance
#  n_steps: number of steps

def runSim(pre_layer,post_layer,spikes):
    torch.cuda.synchronize(torch.device('cuda:0'))
    # Propagate spikes from pre to post neurons
    #start = default_timer()
    const_thr(pre_layer,post_layer)
    #print("add_spikes time: ", default_timer() - start)
    #start = default_timer()
    calc_spikes(post_layer,spikes)
    #print("calc_spikes time: ", default_timer() - start)
    # Calculate STDP weight changes
    #start = default_timer()
    calc_stdp(pre_layer,post_layer,spikes)
    #print("calc_stdp time: ", default_timer() - start)
    # Normalise firing rates and inhibitory balance
    #start = default_timer()
    norm_rates(pre_layer,post_layer)
    #print("norm_rates time: ", default_timer() - start)
    #start = default_timer()
    norm_inhib(post_layer)
    #print("norm_inhib time: ", default_timer() - start)
    #start = default_timer()
    torch.cuda.empty_cache()
    #print("clear cache time: ", default_timer() - start)
def testSim(net,device):
    
    net['step_num'] += 1
    add_spikesTest(net,device,net['spike_dims'])
    for n in range(int(len(net['W'])/2)): # run loop for number of layers
        layers = [int(0 + (n*2)), int(1 + (n*2)), int(0 + n)]
        calc_spikes(net,layers)