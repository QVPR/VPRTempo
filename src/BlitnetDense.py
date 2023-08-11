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
import timeit

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
    #  fast_inhib = fast inhib weights flag
    #  W_lyr      = pre and post layer numbers
    #  eta_stdp   = STDP learning rate (-ve for inhib synapses)
    #
    # ** SIMULATION FIELDS **
    #  step_num = current step

    #pdb.set_trace()
    net = dict(x=[],x_input=[],x_prev=[],x_calc=[],x_fastinp=[],dim=[],thr=[],
               fire_rate=[],have_rate=[],mean_rate=[],eta_ip=[],const_inp=[],nois=[],
               set_spks=[],sspk_idx=[],spikes=[],rec_spks=[],
               W=[],I=[],is_inhib=[],fast_inhib=[],W_lyr=[],eta_stdp=[],
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
    net['mean_rate'].append(torch.zeros(dims[0],dims[1],dims[2],device=device))
    net['eta_ip'].append(ip_rate)
    temp_thr = torch.zeros(dims[0],dims[1],dims[2], device=device)
    net['thr'].append(temp_thr.uniform_(thr_range[0],thr_range[1]))
    temp_fire = torch.zeros(dims[0],dims[1],dims[2], device=device)
    net['fire_rate'].append(temp_fire.uniform_(fire_rate[0],fire_rate[1]))
    net['have_rate'].append(any(net['fire_rate'][-1][0]>0.0))

    temp_const = torch.zeros(dims[0],dims[1],dims[2], device=device)
    net['const_inp'].append(temp_const.uniform_(const_inp[0],const_inp[1]))

    net['nois'].append(nois)
    net['set_spks'].append([])
    net['sspk_idx'].append(0)
    net['spikes'].append(torch.empty([],dtype=torch.float64))
    net['rec_spks'].append(rec_spks)

    return len(net['x'])-1

##################################
# Add a set of random connections between layers
#  net:        BITnet instance
#  layer_pre:  presynaptic layer
#  layer_post: postsynaptic layer
#  W_range:    weight range [lo,hi]
#  p:          initial connection probability
#  stdp_rate:  STDP rate (0=no STDP)
#  fast_inhib: is this fast inhibition (ie inhib applied at same timestep)

def addWeights(net,layer_pre,layer_post,W_range,p,stdp_rate,fast_inhib):

    # get torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    # Check constraints etc
    if np.isscalar(W_range): W_range = [W_range,W_range]
    
    # determine dimensions of the weight matrices
    ncol =net['x'][layer_pre][0].size(dim=0)
    nrow = net['x'][layer_post][0].size(dim=0)
    
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
            Iindex = len(net['I']) - 1
            net['I'].append(torch.zeros(nrow, device=device))
            net['I'][Iindex] = torch.unsqueeze(net['I'][Iindex],0)
            # append single reference arguments
            net['W_lyr'].append([layer_pre,layer_post])
            net['eta_stdp'].append(stdp_rate)
            net['is_inhib'].append(W_range[0]<0.0 and W_range[1]<=0.0)
            net['fast_inhib'].append(fast_inhib and net['is_inhib'][-1])
        else:
            net['I'][Iindex] = torch.concat((net['I'][Iindex],
                                torch.unsqueeze(torch.zeros(nrow, device=device),0)),0)
        
        # remove connections based on calculated indexes
        net['W'][excIndex][n,:,:][setzeroExc] = 0.0 # excitatory connections
        net['W'][inhIndex][n,:,:][setzeroInh] = 0.0 # inhibitory connections
    
        # Normalise the weights (except fast inhib weights)
        if not net['fast_inhib'][-1]:
            nrmExc = torch.linalg.norm(net['W'][excIndex][len(net['W'][excIndex])-1],ord=1,axis=0)
            nrmInh = torch.linalg.norm(net['W'][inhIndex][len(net['W'][inhIndex])-1],ord=1,axis=0)
            nrmExc[nrmExc==0.0] = 1.0
            nrmInh[nrmInh==0.0] = 1.0
            net['W'][excIndex][n] = net['W'][excIndex][n,:,:]/nrmExc
            net['W'][inhIndex][n] = net['W'][inhIndex][n,:,:]/nrmInh

    return len(net['W'])-1

##################################
# Set defined spike times for a neuron layer (ie a neuron population)
#  net:   BITnet instance
#  layer: layer number
#  times: 2-column matrix (col 1 = step num (ordered); col 2 = neuron num to spike)
# NOTE for spike forcing an output layer ensure that: eta_ip=0 and target fire_rate=0
# FOLLOWING training ensure that: forced spikes array is removed, ie: setSpikeTimes(n,l,[])

def setSpikeTimes(net,layer,times):
    if isinstance(times,list):
        net['set_spks'][layer] = times.copy()
    else:    
        net['set_spks'][layer] = times.detach().clone()
    net['sspk_idx'][layer] = 0
    
##################################
# Normalise all the firing rates
#  net: BITnet instance

def norm_rates(net):

    for i,rate in enumerate(net['fire_rate']):
        if rate.any() and net['eta_ip'][i] > 0.0:
            net['thr'][i] = net['thr'][i] + net['eta_ip'][i]*(net['x'][i]-rate)
            #xxx net['thr'][i] = net['thr'][i] + net['eta_ip'][i]*(net['x'][i]-rate)
            net['thr'][i][net['thr'][i]<0.0] = 0.0 #xxx
            
##################################
# Normalise inhib weights to balance input currents
#  net: BITnet instance

def norm_inhib(net):

    #return #xxx no norm_inhib
    for i,W in enumerate(net['W']):
        if net['eta_stdp'][i] < 0: # and not net['fast_inhib'][i]:
        #if net['is_inhib'][i]: # and not net['fast_inhib'][i]:
            lyr = net['W_lyr'][i][1]
            #wadj = np.multiply(W,np.sign(net['x_input'][lyr]))*-net['eta_stdp'][i]*10
            #wadj = np.multiply(W,net['x_input'][lyr]-net['fire_rate'][lyr])*-net['eta_stdp'][i]*100
            try:
                wadj = np.multiply(W,net['x_input'][lyr])*-net['eta_stdp'][i]*50 #0.5 #100
                net['W'][i] += wadj
                net['W'][i][W>0.0] = -0.000001
            except RuntimeWarning:
                print("norm_inhib err")
                pdb.set_trace()
                
##################################
# Propagate spikes thru the network
#  net: SORN instance

def calc_spikes(net,device):
    # Start with the constant input in the neurons of each layer
    for i,const in enumerate(net['const_inp']):
        net['x_input'][i] = torch.full_like(net['x_input'][i],0.0)
        net['x_input'][i] += net['const_inp'][i]
        # Find the threshold crossings (overwritten later if needed)
        net['x'][i] = torch.clamp((net['x_input'][i]-net['thr'][i]),0.0,0.9)
    
    # insert any predefined spikes
    start = net['spike_dims'] * (net['step_num'] - 1)
    end = net['spike_dims'] * net['step_num']
    for i in range(len(net['set_spks'])):
        if len(net['set_spks'][i]):
            index = torch.arange(start,end,device=device,dtype=int)
            net['x'][i] = torch.index_select(net['set_spks'][i],1,index)
    
    # get layer index information
    excW = len(net['W'])-1
    inhW = len(net['W'])
    layers = net['W_lyr'][len(net['W_lyr'])-1]

    # Synaptic currents last for 1 timestep
    # excitatory weights
    net['I'][layers[0]] = torch.bmm(net['W'][excW-1],net['x'][layers[0]])
    net['x_input'][layers[1]] += net['I'][layers[0]]
    #inhibitory weights
    net['I'][layers[0]] = torch.bmm(net['W'][inhW-1],net['x'][layers[0]])
    net['x_input'][layers[1]] += net['I'][layers[0]]

    # Find threshold crossings
    if layers[0]!=layers[1]:
        net['x_prev'][j] = net['x'][j][:]
    if not len(net['set_spks'][j]):
        # No predefined spikes for this layer
        net['x'][j] = np.clip(net['x_input'][j]-net['thr'][j],a_min=0.0,a_max=0.9)
    else:
        # Predefined spikes exist for this layer, remember the calculated ones
        net['x_calc'][j] = np.clip(net['x_input'][j]-net['thr'][j],a_min=0.0,a_max=0.9)
    if layers[0]==layers[1]:
        net['x_prev'][j] = net['x'][j][:]
    # If the next weight matrix is fast inhib for this layer, process it now
    if i < len(net['W'])-1:
        if net['fast_inhib'][i+1] and layers[1]==net['W_lyr'][i+1][1]:
            flyrs = net['W_lyr'][i+1]
            net['x_fastinp'][flyrs[1]] = net['x_input'][flyrs[1]].copy()
            if flyrs[0]==flyrs[1]:
                postsyn_spks = np.tile(net['x'][flyrs[0]],[len(net['x'][flyrs[0]]),1])
                presyn_spks = np.transpose(postsyn_spks)
                presyn_spks[presyn_spks < postsyn_spks] = 0.0
                net['x_fastinp'][flyrs[1]] += np.sum((presyn_spks)*net['W'][i+1],0)
            else:
                net['x_fastinp'][flyrs[1]] += np.matmul(net['x'][flyrs[0]],net['W'][i+1])
            if not len(net['set_spks'][j]):
                # No predefined spikes for this layer
                net['x'][flyrs[1]] = np.clip(net['x_fastinp'][flyrs[1]]-net['thr'][flyrs[1]],
                                         a_min=0.0,a_max=0.9)
            else:
                # Predefined spikes exist for this layer, remember the calculated ones
                net['x_calc'][flyrs[1]] = np.clip(net['x_fastinp'][flyrs[1]]-net['thr'][flyrs[1]],
                                         a_min=0.0,a_max=0.9)                 
                
    # Finally, update mean firing rates and record all spikes if needed
    for i,eta in enumerate(net['eta_ip']):
        
        if eta > 0.0:
            net['mean_rate'][i] = net['mean_rate'][i]*(1.0-eta) +\
                                  (net['x'][i]>0.0)*eta
        if net['rec_spks'][i]:
            outspk = (net['x'][i]).detach().cpu().numpy()
            if i == 2:
                outspk[outspk<0.05] = 0
            n_idx = np.nonzero(outspk)
            net['spikes'][i].extend([net['step_num']+net['x'][i][n].detach().cpu().numpy(),n]
                                        for n in n_idx)

##################################
# Calculate STDP
#  net: BITnet instance

def calc_stdp(net):

    # Loop thru weight matrices that have non-zero learning rate
    for i,W in enumerate(net['W']):
        if net['eta_stdp'][i] != 0:

            # Remember layer numbers and weight matrix shape
            layers = net['W_lyr'][i]
            shape = W.size()

            #
            # Spike Forcing has special rules to make calculated and forced spikes match
            #
            if len(net['set_spks'][layers[1]]):

                # Diff between forced and calculated spikes
                xdiff = net['x'][layers[1]] - net['x_calc'][layers[1]]
                # Modulate learning rate by firing rate (low firing rate = high learning rate)
                #if net['have_rate'][layers[1]]:
                #    xdiff /= net['fire_rate'][layers[1]]
                    
                # Threshold rules - lower it if calced spike is smaller (and vice versa)
                net['thr'][layers[1]] -= np.sign(xdiff)*np.abs(net['eta_stdp'][i])/10
                net['thr'][layers[1]][net['thr'][layers[1]]<0.0] = 0.0  # don't go -ve
                
                # A little bit of threshold decay
                #net['thr'][layers[1]] *= (1-net['eta_stdp'][i]/100)

                # Pre and Post spikes tiled across and down for all synapses
                if net['have_rate'][layers[0]]:
                    # Modulate learning rate by firing rate (low firing rate = high learning rate)
                    mpre = net['x'][layers[0]]/net['fire_rate'][layers[0]]
                else:
                    mpre = net['x'][layers[0]]
                pre  = torch.from_numpy(np.tile(np.reshape(mpre, [shape[0],1]),[1,shape[1]]))
                post = torch.from_numpy(np.tile(np.reshape(xdiff,[1,shape[1]]),[shape[0],1]))

                # Excitatory connections
                if net['eta_stdp'][i] > 0:
                    havconn = W>0
                    inc_stdp = pre*post*havconn
                # Inhibitory connections
                else:
                    havconn = W<0
                    inc_stdp = -pre*post*havconn

                # Apply the weight changes
                net['W'][i] += inc_stdp*net['eta_stdp'][i]

            #
            # Normal STDP
            #
            elif not net['fast_inhib'][i]:

                pre = torch.from_numpy(np.tile(np.reshape(net['x'][layers[0]],[shape[0],1]),[1,shape[1]]))
                if net['have_rate'][layers[1]]:
                    # Modulate learning rate by firing rate (low firing rate = high learning rate)
                    mpost = net['x'][layers[1]] #/net['fire_rate'][layers[1]]
                else:
                    mpost = net['x'][layers[1]]
                post = torch.from_numpy(np.tile(np.reshape(mpost,[1,shape[1]]),[shape[0],1]))

                # Excitatory synapses
                if net['eta_stdp'][i] > 0:
                    havconn = W>0
                    inc_stdp = (0.5-post)*(pre>0)*(post>0)*havconn
                # Inhibitory synapses
                elif not net['fast_inhib'][i]: # and False:
                    havconn = W<0
                    inc_stdp = (0.5-post)*(pre>0)*(post>0)*havconn

                # Apply the weight changes
                net['W'][i] += inc_stdp*net['eta_stdp'][i]

            #
            # Fast inhibitory synapses, xxx update for firing rate modulation of eta_stdp?
            #
            else:

                # Store weight changes
                inc_stdp = np.zeros(shape)
                dec_stdp = np.zeros(shape)

                # Loop thru firing pre neurons
                for pre in np.where(net['x'][layers[0]])[0]:
                    # Loop thru ALL post neurons
                    for post in range(len(net['x'][layers[1]])):
                        if net['W'][i][pre,post]!=0:
                            if net['x'][layers[1]][post] > 0.0:
                                if net['x'][layers[0]][pre] >\
                                   net['x'][layers[1]][post]:
                                    # Synapse gets stronger if pre fires before post
                                    inc_stdp[pre,post] = 0.5 #0.1 #/\
                                                   #net['mean_rate'][layers[1]][post]
                                                   #net['mean_rate'][layers[0]][pre]
                                else:
                                    # Synapse gets weaker if pre fires after post
                                    dec_stdp[pre,post] = 0.5 *\
                                            (1.0-net['mean_rate'][layers[1]][post])
                            else:
                                # Also gets weaker if pre fires and not post
                                dec_stdp[pre,post] = 0.5*\
                                                    net['mean_rate'][layers[1]][post]

                # Apply the weight changes
                net['W'][i] += inc_stdp*net['eta_stdp'][i]
                net['W'][i] -= dec_stdp*net['eta_stdp'][i]

            #
            # Finish
            #
            
            # Try weight decay?
            #net['W'][i] = net['W'][i]-net['eta_stdp'][i]/10 # * (1-net['eta_stdp'][i])
            if net['eta_stdp'][i] > 0.0:
                # Excitation - pruning and synaptogenesis (structural plasticity)
                net['W'][i][W<0.0] = 0.000001 #xxx
                net['W'][i][W>10.0] = 10.0    #xxx
                if np.random.rand() < 0.0: #0.1: #xxx TEMPORARILY OFF
                    synap = (np.random.rand(2)*shape).astype(int)
                    if net['W'][i][synap[0]][synap[1]] == 0:
                        net['W'][i][synap[0]][synap[1]] = 0.001
            else:
                # Inhibition - must not go +ve
                net['W'][i][W>0.0] = -0.000001 #xxx
                net['W'][i][W<-10.0] = -10.0   #xxx

    # Finally clear out any predefined spikes that are used up (so calculated network spikes can take over)
    for i in range(len(net['set_spks'])):
        if len(net['set_spks'][i]):
            if len(net['set_spks'][i]) <= net['sspk_idx'][i]:
                net['set_spks'][i] = []

##################################
# Run the simulation
#  net: BITnet instance
#  n_steps: number of steps

def runSim(net,n_steps,device):

    # Loop
    for step in range(n_steps):

        # Inc step count
        net['step_num'] += 1

        # Propagate spikes from pre to post neurons
        calc_spikes(net,device)
        
        # Calculate STDP weight changes
        calc_stdp(net)
        
        # Normalise firing rates and inhibitory balance
        norm_rates(net)
        norm_inhib(net)

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
