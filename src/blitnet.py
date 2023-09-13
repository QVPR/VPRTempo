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

import matplotlib.pyplot as plt


##################################
# Return a new empty BITnet instance

def newNet(modules, dims):

    np.random.seed() # new random seed

    # ** NEURON FIELDS **
    #  x         = activations
    #  x_input   = total inputs
    #  x_prev    = previous activations
    #  x_calc    = calculated activations
    #  x_fastinp = total inputs including fast inhib
    #  dim       = dimensions
    #  thr       = thresholds for each neuron
    #  fire_rate = target firing rate for each neuron
    #  have_rate = have a target firing rate
    #  mean_rate = running avg firing rate for each neuron
    #  eta_ip    = IP (threshold) learning rate
    #  const_inp = constant input to each neuron
    #  nois      = noise st.dev.
    #  set_spks  = pre-defined spike times (if any)
    #  sspk_idx  = current index into set_spks
    #  spikes    = spike events
    #  rec_spks  = record spikes?
    #
    # ** CONNECTION FIELDS **
    #  W          = weights (-ve for inhib synapses)
    #  I          = synaptic currents
    #  is_inhib   = inhib weights flag
    #  W_lyr      = pre and post layer numbers
    #  eta_stdp   = STDP learning rate (-ve for inhib synapses)
    #
    # ** SIMULATION FIELDS **
    #  step_num = current step

    #pdb.set_trace()
    net = dict(x=[],x_input=[],x_prev=[],x_calc=[],x_fastinp=[],dim=[],thr=[],
               fire_rate=[],have_rate=[],mean_rate=[],eta_ip=[],const_inp=[],nois=[],
               set_spks=[],sspk_idx=[],spikes=[],rec_spks=[],
               W=[],I=[],is_inhib=[],W_lyr=[],eta_stdp=[],
               step_num=0, num_modules = modules, spike_dims = dims)
    
    return net

##################################
# Add a neuron layer (ie a neuron population)
#  net:       BITnet instance
#  dim:       layer dimensions [x,y,...]
#  thr_range: initial threshold range
#  fire_rate: target firing rate (0=no target)
#  ip_rate:   intrinsic threshold plasticity (IP) rate (0=no IP)
#  const_inp: constant input to each neuron (0=none)
#  nois:      noise variance (0=no noise)
#  rec_spks:  record spikes?

def addLayer(net,dims,thr_range,fire_rate,ip_rate,const_inp,nois,rec_spks):
    
    # get torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Check constraints etc
    if np.isscalar(thr_range): thr_range = [thr_range,thr_range]
    if np.isscalar(fire_rate): fire_rate = [fire_rate,fire_rate]
    if np.isscalar(const_inp): const_inp = [const_inp,const_inp]
    
    # create layer tensors
    net['dim'].append(np.array(dims,int))
    net['x'].append(torch.zeros(dims[0],dims[1],dims[2],device=device))
    net['x_prev'].append(torch.zeros(dims[0],dims[1],dims[2],device=device))
    net['x_calc'].append(torch.zeros(dims[0],dims[1],dims[2],device=device))
    net['x_input'].append(torch.zeros(dims[0],dims[1],dims[2],device=device))
    net['x_fastinp'].append(torch.zeros(dims[0],dims[1],dims[2],device=device))
    net['eta_ip'].append(ip_rate)
    temp_thr = torch.zeros(dims[0],dims[1],dims[2], device=device)
    net['thr'].append(temp_thr.uniform_(thr_range[0],thr_range[1]))
    temp_fire = torch.zeros(dims[0],dims[1],dims[2], device=device)
    net['fire_rate'].append(temp_fire.uniform_(fire_rate[0],fire_rate[1]))
    net['have_rate'].append(any(net['fire_rate'][-1][:,:,0]>0.0))

    temp_const = torch.zeros(dims[0],dims[1],dims[2], device=device)
    net['const_inp'].append(temp_const.uniform_(const_inp[0],const_inp[1]))

    net['nois'].append(nois)
    net['set_spks'].append([])
    net['sspk_idx'].append(0)
    net['spikes'].append(torch.empty([],dtype=torch.float64))
    net['rec_spks'].append(rec_spks)

##################################
# Add a set of random connections between layers
#  net:        BITnet instance
#  layer_pre:  presynaptic layer
#  layer_post: postsynaptic layer
#  W_range:    weight range [lo,hi]
#  p:          initial connection probability
#  stdp_rate:  STDP rate (0=no STDP)

def addWeights(net,layer_pre,layer_post,W_range,p,stdp_rate):

    # get torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    # Check constraints etc
    if np.isscalar(W_range): W_range = [W_range,W_range]
    
    # determine dimensions of the weight matrices
    nrow =net['x'][layer_pre].size(dim=2)
    ncol = net['x'][layer_post].size(dim=2)
    
    # calculate mean and std for normal distributions
    inWmn = (W_range[0]+W_range[1])/2.0
    inWsd = (W_range[1]-W_range[0])/6.0
    exWmn = (W_range[1]+W_range[2])/2.0
    exWsd = (W_range[2]-W_range[1])/6.0
    
    # loop through modules and add excitatory and inhibitory weights
    for n in range(net['num_modules']):
        if n == 0: # first weights to be appended
            net['W'].append(torch.empty(nrow,ncol,device=device).normal_(mean=exWmn,std=exWsd)) # excitatory weights
            excIndex = len(net['W']) - 1
            net['W'][excIndex] = torch.unsqueeze(net['W'][excIndex],0)
            net['W'].append(torch.empty(nrow,ncol,device=device).normal_(mean=inWmn,std=inWsd)) # inhibitory weights
            inhIndex = len(net['W']) - 1
            net['W'][inhIndex] = torch.unsqueeze(net['W'][inhIndex],0)
        else: # stack new weights onto appended weight
            net['W'][excIndex] = torch.concat((net['W'][excIndex], 
                    torch.unsqueeze(torch.empty(nrow,ncol,device=device).normal_(mean=exWmn,std=exWsd),0)),0)
            net['W'][inhIndex] = torch.concat((net['W'][inhIndex], 
                    torch.unsqueeze(torch.empty(nrow,ncol,device=device).normal_(mean=inWmn,std=inWsd),0)),0) # inhibitory weights
        net['W'][excIndex][n][net['W'][excIndex][n] < 0] = 0.0 # remove -ve excitatory weights
        net['W'][inhIndex][n][net['W'][inhIndex][n] > 0] = 0.0 # remove +ve inhibitory weights
        
        # remove connections based on exc and inh probabilities
        setzeroExc = np.random.rand(nrow,ncol) > p[0]
        setzeroInh = np.random.rand(nrow,ncol) > p[1]
        
        # add current
        if n == 0:
            Iindex = len(net['I'])
            net['I'].append(torch.zeros(nrow, device=device))
            net['I'][Iindex] = torch.unsqueeze(net['I'][Iindex],0)
            # append single reference arguments
            net['W_lyr'].append([layer_pre,layer_post])
            net['eta_stdp'].append(stdp_rate)
            net['eta_stdp'].append(-stdp_rate)
            net['is_inhib'].append(W_range[0]<0.0 and W_range[1]<=0.0)
        else:
            net['I'][Iindex] = torch.concat((net['I'][Iindex],
                                torch.unsqueeze(torch.zeros(nrow, device=device),0)),0)
        
        # remove connections based on calculated indexes
        if setzeroExc.any():
            net['W'][excIndex][n,:,:][setzeroExc] = 0.0 # excitatory connections
        if setzeroInh.any():
            net['W'][inhIndex][n,:,:][setzeroInh] = 0.0 # inhibitory connections
    
        # Normalise the weights (except fast inhib weights)
        nrmExc = torch.linalg.norm(net['W'][excIndex][len(net['W'][excIndex])-1],ord=1,axis=0)
        nrmInh = torch.linalg.norm(net['W'][inhIndex][len(net['W'][inhIndex])-1],ord=1,axis=0)
        nrmExc[nrmExc==0.0] = 1.0
        nrmInh[nrmInh==0.0] = 1.0
        net['W'][excIndex][n] = net['W'][excIndex][n,:,:]/nrmExc
        net['W'][inhIndex][n] = net['W'][inhIndex][n,:,:]/nrmInh
    
##################################
# Normalise all the firing rates
#  net: BITnet instance

def norm_rates(net):

    for i,rate in enumerate(net['fire_rate']):
        if rate.any() and net['eta_ip'][i] > 0.0:
            net['thr'][i] = net['thr'][i] + net['eta_ip'][i]*(net['x'][i]-rate)
            net['thr'][i][net['thr'][i]<0.0] = 0.0
            
##################################
# Normalise inhib weights to balance input currents
#  net: BITnet instance

def norm_inhib(net):

    for i,W in enumerate(net['W']):
        if net['eta_stdp'][i] < 0:
            lyr = net['W_lyr'][len(net['W_lyr'])-1]
            try:
                wadj = (net['x_input'][lyr[1]]*W)*-net['eta_stdp'][i]*50
                net['W'][i] += wadj
                net['W'][i][W>0.0] = -0.000001
            except RuntimeWarning:
                print("norm_inhib err")
                pdb.set_trace()
                
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



def add_spikes(net,device,dims):
    
    # Start with the constant input in the neurons of each layer
    for i,const in enumerate(net['const_inp']):
        if len(net['W_lyr']) > 1 and i == 1:
            net['x_input'][i] = net['x_feat'][net['step_num']-1]
        else:
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

def calc_spikes(net,layersInfo):  

    # get layer index information
    excW = layersInfo[0]
    inhW = layersInfo[1]
    layers = net['W_lyr'][layersInfo[2]]

    # Synaptic currents last for 1 timestep
    # excitatory weights
    net['I'][layers[0]] = torch.bmm(net['x'][layers[0]],net['W'][excW])
    net['x_input'][layers[1]] += net['I'][layers[0]]
    #inhibitory weights
    net['I'][layers[0]] = torch.bmm(net['x'][layers[0]],net['W'][inhW])
    net['x_input'][layers[1]] += net['I'][layers[0]]
    
    # adjust the spikes based on threshold
    if not len(net['set_spks'][layers[1]]): # No predefined spikes for this layer
        net['x'][layers[1]] = torch.clamp((net['x_input'][layers[1]]-net['thr'][layers[1]]),
                                          min=0.0,max=0.9)
    else: # Predefined spikes exist for this layer, remember the calculated ones
        net['x_calc'][layers[1]] = torch.clamp((net['x_input'][layers[1]]-net['thr'][layers[1]]),
                                        min=0.0,max=0.9)
    
    # update the x previous variable with calculated spikes
    net['x_prev'][layers[1]] = net['x'][layers[1]]              
                
    # Finally, update mean firing rates and record all spikes if needed
    for i,eta in enumerate(net['eta_ip']):
        
        if net['rec_spks'][i]:
            outspk = (net['x'][i][0,0,:]).detach().cpu().numpy() # detach to numpy for easy plotting
            n_idx = np.nonzero(outspk)
            net['spikes'][i].extend([net['step_num']+net['x'][i][0,:,:].detach().cpu().numpy(),n]
                                        for n in n_idx)

##################################
# Calculate STDP
#  net: BITnet instance

def calc_stdp(net,layerInfo):
    
    # get weight layer information
    excW = layerInfo[0]
    inhW = layerInfo[1]
    layers = net['W_lyr'][layerInfo[2]]
    shape = list(net['W'][excW].size())

    #
    # Spike Forcing has special rules to make calculated and forced spikes match
    #
    if len(net['set_spks'][layers[1]]): # will run for the output layer

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
        
        # tile out pre- and post-spikes
        pre = torch.tile(torch.reshape(net['x'][layers[0]],(shape[0],shape[1],1)),(1,shape[2]))
        post = torch.tile(net['x'][layers[1]],(shape[1],1))

        # Excitatory synapses
        havconnExc = net['W'][excW]>0
        inc_stdpExc = (0.5-post)*(pre>0)*(post>0)*havconnExc
        # Inhibitory synapses
        havconnInh = net['W'][inhW]<0
        inc_stdpInh = (0.5-post)*(pre>0)*(post>0)*havconnInh

        # Apply the weight changes
        net['W'][excW] += inc_stdpExc*net['eta_stdp'][excW]
        net['W'][inhW] += inc_stdpInh*net['eta_stdp'][inhW]
        
    # fix connection weights if too +ve or -ve
    # Excitation - pruning and synaptogenesis (structural plasticity)
    net['W'][excW][net['W'][excW]<0.0] = 0.000001
    net['W'][excW][net['W'][excW]>10.0] = 10.0
    # Inhibition - must not go +ve
    net['W'][inhW][net['W'][inhW]>0.0] = -0.000001
    net['W'][inhW][net['W'][inhW]<-10.0] = -10.0

##################################
# Run the simulation
#  net: BITnet instance
#  n_steps: number of steps

def runSim(net,n_steps,device,layers):

    # Inc step count
    net['step_num'] += 1
    # Propagate spikes from pre to post neurons
    add_spikes(net,device,net['spike_dims'])
    calc_spikes(net,layers)
    
    # Calculate STDP weight changes
    calc_stdp(net,layers)
    
    # Normalise firing rates and inhibitory balance
    norm_rates(net)
    norm_inhib(net)
        
def testSim(net,device):
    
    net['step_num'] += 1
    add_spikesTest(net,device,net['spike_dims'])
    for n in range(int(len(net['W'])/2)): # run loop for number of layers
        layers = [int(0 + (n*2)), int(1 + (n*2)), int(0 + n)]
        calc_spikes(net,layers)

##################################
# Plot recorded spikes in current subplot
#  net: BITnet instance

def subplotSpikes(net,cutoff):
     
    n_tot = 0
    for i,sp in enumerate(net['spikes']):
        x=[]; y=[]
        for n in sp:
            x.extend(list(n[0]))
            y.extend(list(n[1]+n_tot))

        plt.plot(x,y,'.',ms=1)
        n_tot += np.size(net['x'][i].detach().cpu().numpy())
    
##################################
# Plot recorded spikes in new figure
#  net: BITnet instance

def plotSpikes(net,cutoff):

    plt.figure()
    subplotSpikes(net,cutoff)
    plt.show(block=False)
    
##################################
