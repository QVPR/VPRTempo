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
import torch


def addWeights(W_range=[-1,0,1],p=[1,1],dims=None,num_modules=1):

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
                  0)), 0)
            # inhibitory weights
            inhW = torch.cat((inhW, 
                  torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=inWmn, std=inWsd), 
                  0)), 0)
        else:  # stack new weights onto appended weight
            excW = torch.cat((excW, 
                  torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=exWmn, std=exWsd), 
                  0)), 0)
            inhW = torch.cat((inhW, 
                  torch.unsqueeze(torch.empty(nrow, ncol, device=device).normal_(mean=inWmn, std=inWsd), 
                  0)), 0)
        
        # Remove negative excitatory weights
        excW[n][excW[n] < 0] = 0.0
        # Remove positive inhibitory weights
        inhW[n][inhW[n] > 0] = 0.0

        # remove connections based on exc and inh probabilities
        setzeroExc = np.random.rand(nrow,ncol) > p[0]
        setzeroInh = np.random.rand(nrow,ncol) > p[1]
        
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
        
    return excW, inhW, havconnExc, havconnInh

def calc_spikes(post_layer, spikes):  
    
    # Add the constant input
    post_layer.x_input += post_layer.const_inp
    
    # Multiply input spikes by positive and negative weights
    post_layer.x_input += torch.bmm(spikes, post_layer.excW.detach())
    post_layer.x_input += torch.bmm(spikes, post_layer.inhW.detach())
    
    # Clamp outputs between 0 and 0.9 after subtracting thresholds from input
    if post_layer.spk_force: 
        post_layer.x_calc = torch.clamp(torch.sub(post_layer.x_input, post_layer.thr.detach()),
                                                    min=0.0, max=0.9)
    else: 
        post_layer.x = torch.clamp(torch.sub(post_layer.x_input, post_layer.thr.detach()), 
                                                min=0.0, max=0.9)

def calc_stdp(pre_layer, post_layer, spikes, idx=0):

    # Spike Forcing has special rules to make calculated and forced spikes match
    if post_layer.spk_force:
        
        # Get layer dimensions
        shape = [len(post_layer.excW[:, 0, 0]),
                 len(post_layer.excW[0, :, 0]),
                 len(post_layer.excW[0, 0, :])]
        # Get the output neuron index
        idx_sel = torch.arange(int(idx[0]), int(idx[0]) + 1, device=post_layer.device, dtype=int)

        # Difference between forced and calculated spikes
        post_layer.x = torch.full_like(post_layer.x, 0)
        xdiff = torch.clamp(post_layer.x.index_fill_(-1, idx_sel, 0.5) - post_layer.x_calc,
                            min=0, max=1)

        # Threshold rules - lower it if calced spike is smaller (and vice versa)
        post_layer.thr.data -= torch.sign(xdiff) * torch.abs(post_layer.eta_stdp) / 10
        post_layer.thr.data -= torch.sign(xdiff) * torch.abs((post_layer.eta_stdp * -1)) / 10
        post_layer.thr.data = post_layer.thr.data.clamp(min=0, max=1)

        # Pre and Post spikes tiled across and down for all synapses
        if pre_layer.have_rate:
            # Modulate learning rate by firing rate (low firing rate = high learning rate)
            mpre = spikes / pre_layer.fire_rate
        else:
            mpre = spikes
        pre = torch.tile(torch.reshape(mpre, (shape[0], shape[1], 1)), (1, shape[2]))
        post = torch.tile(xdiff, (shape[1], 1))

        # Apply the weight changes
        post_layer.excW.data += (pre * post * post_layer.havconnExc) * post_layer.eta_stdp

        post_layer.inhW.data += (-pre * post * post_layer.havconnInh) * (post_layer.eta_stdp * -1)

    # Normal STDP
    else:
        
        # Get layer dimensions
        shape = [len(post_layer.excW[:, 0, 0]),
                 len(post_layer.excW[0, :, 0]),
                 len(post_layer.excW[0, 0, :])]

        # Tile out pre- and post-spikes
        pre = torch.tile(torch.reshape(spikes, (shape[0], shape[1], 1)), (1, shape[2]))
        post = torch.tile(post_layer.x, (shape[1], 1))
        
        # Apply positive and negative weight changes
        post_layer.excW.data += ((0.5 - post) * (pre > 0) * (post > 0) * 
                                 post_layer.havconnExc) * post_layer.eta_stdp
        post_layer.inhW.data += ((0.5 - post) * (pre > 0) * 
                                 (post > 0) * post_layer.havconnInh) * (post_layer.eta_stdp * -1)

    # In-place clamp for excitatory and inhibitory weights
    post_layer.excW.data[post_layer.excW.data < 0] = 1e-06
    post_layer.inhW.data[post_layer.inhW.data > 0] = -1e-06
    
    # Remove negative weights for excW and positive for inhW
    post_layer.excW.data[post_layer.havconnExc] = post_layer.excW.data[post_layer.havconnExc].clamp(min=1e-06, max=10)
    post_layer.inhW.data[post_layer.havconnInh] = post_layer.inhW.data[post_layer.havconnInh].clamp(min=-10, max=-1e-06)
    
def norm_rates(post_layer):
    
    # Check if layer has target firing rate and an ITP learning rate
    if post_layer.have_rate and post_layer.eta_ip > 0.0:
        
        # Replace the original layer.thr with the updated one
        post_layer.thr.data += post_layer.eta_ip * (post_layer.x - post_layer.fire_rate)
        post_layer.thr.data[post_layer.thr.data < 0] = 0

def norm_inhib(post_layer):
    
    # Check if layer has inhibitory weights and an stdp learning rate
    if torch.any(post_layer.inhW).item() and post_layer.eta_stdp != 0:
        
        # Normalize the inhibitory weights using homeostasis
        post_layer.inhW.data += torch.mul(torch.mul(post_layer.x_input, post_layer.inhW.data), 
                                           post_layer.eta_stdp*50)
        post_layer.inhW.data[post_layer.inhW.data > 0.0] = -1e-06
        
def runSim(pre_layer,post_layer,spikes,idx):

    # Propagate spikes from pre to post neurons
    calc_spikes(post_layer,spikes)

    # Calculate STDP weight changes
    calc_stdp(pre_layer,post_layer,spikes,idx)

    # Normalise firing rates and inhibitory balance
    norm_rates(post_layer)
    norm_inhib(post_layer)

def testSim(layers,layer_num,spikes):
    
    # run the test system through all specified layers to get an output
    for n in range(layer_num):
        layers['layer'+str(n+1)].x.fill_(0.0)
        layers['layer'+str(n+1)].x_input.fill_(0.0)
        calc_spikes(layers['layer'+str(n+1)],spikes)
        spikes=layers['layer'+str(n+1)].x
        
    return spikes