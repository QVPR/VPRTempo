##################################
#
# BLiTnet implementation (Binned Linear-Time networks)
# ...previously
# BITnet implementation (BInary-neuron binned-Time networks)
#
# Changes for version 09:
# - Re-introduced explicit depression for normal inhib
# connections rather than relying on normalisation to implcitly cause
# depression.
# - Changed the inc_stdp amount for inhib STDP to be
# normalised based on the firing rate of the post neuron not the pre,
# since the pre neuron firing is the baseline presumption for all
# inhib STDP (ie pre->post = inc inhib, pre->no_post_spike = dec
# inhib), so to ensure that independent activity leads to exactly 0
# inhib weight means normalising on the post rate.
# - Removed erroneous normalisation of the depression side of the
# lateral inhib - was causing too little depression and the lateral
# inhib was building up too strong
#
# Changes for version 10:
# - for fast inhib, don't inhib if time diff = 0 (calc_spikes)
# - code for init weights as normal rather than uniform (addWeights)
# - add x_fastinp field to keep fast inhib separate from inhib normalisation (calc_spikes)
#
# Changes for version 11:
# - allowed set_spikes to set spike time with sub-timestep resolution (setSpikeTimes)
#
# Changes for version 12:
# - fast inhib spikes are all propagated with value 1 (calc_spikes)
#
# Changes for version 13:
# - improved weight decay, but currently disabled (calc_stdp)
# - used tiled matrices rather than loops for STDP calcs (calc_stdp)
# - removed the spike shift of +0.5, but may put it back (calc_spikes)
#
# Changes for version 14:
# - changed spike forcing - no unforced spikes allowed and added specific new STDP rules
#   (calc_spikes and calc_stdp)
#
# Changes for version 15:
# - don't prune weights that hit zero, just set them very small (1e-6 or -1e6) (calc_stdp)
# - new function to plot spikes in a subplot (subplotSpikes)
#
# Changes for version 16:
# - reinstated the spike shift of +0.5 that was removed in version 13 (calc_spikes)
#
# Changes for version 17:
# - re-removed the spike shift of +0.5 (calc_spikes); it helped signal propagation but
#   hindered feature extraction; instead, make weights stronger
# - added wght_norm for weights (the total weight value to normalize to)
# 
# Changes for version 18:
# - changed max spike amplitude from 1.0 to 0.9 (to avoid confusion with next timestep)
#
# Changes for version 19:
# - BIG change to propagate spikes up thru a hierarchy in 1 step (calc_spikes, calc_stdp)
#
# Changes for version 20:
# - STDP was broken in version 19 - fixed
#
# Changes for version 21:
# - BIG change to remove normalisation of exc weights to a fixed value, instead normalise
#   so spike amplitude averages 0.5; gets rid of normalisation 'fudge factor' and creates
#   homeostasis between ITP (intrinsic threshold plasticity) and spike amplitudes (weights)
# - Also limit any synapse to max 1.0, and fixed a missing STDP component of spike forcing
#
# Changes for version 21, PART B:
# - BIG change to STDP which still wasn't working the best, see Chap 14 of full report
# - also increased learning rate for inhib normalisation which I set too low in prev version
#
# Changes for version 22:
# - totally rewrote spike forcing, can now easily outperform linear decoding
# - removed dec_stdp term from normal STDP (now just relies on the (0.5-post) term)
# - removed wght_norm field and norm_weights() function (no longer used)
# - removed all spike 'shift' code (also no longer needed)
# - removed limit of +/-1 on all weights which was sometimes upsetting balance
#
# Changes for version 23:
# - ensure weights stay negative during inhib weight normalisation (norm_inhib)
# - re-instate weight limits but this time set max. weight to +/-10
#
# Changes for version 24:
# - modulate STDP learning rate by firing rate (low firing rate = high learning rate needed)
#
# Changes for version 25:
# - clear predefined spikes when they are all used up
# - fixed 'x_prev' to work for recurrent connections (calc_spikes)
#
#
# SOME IDEAS:
# - build-in learning rate annealing (pass: start rate, num timesteps, annealing power)
# - build-in analog inputs (pass: min and max value, num input neurons, bump radius)
# - allow thresholds to go -ve to allow spikes to occur from removal of tonic inhibition
# - calc_stdp still needs work for recurrent connections
#
#
# FULL REPORTS (latest first):
# https://docs.google.com/document/d/19uybCShRo5kUJXRgVOEnH5GoLQy-RgeP7cqhvw7XYwM
# and
# https://docs.google.com/document/d/17H_dOlwNNLmLUdHQfqvUQ8cC0IBmLunbq-CYumH8vMY
# and
# https://docs.google.com/document/d/1JcQHpmTQNsP9a2tcIwsjg_Fwj-2FSAmOAv30uWOg7mA
#
##################################

import numpy as np
import matplotlib.pyplot as plt
import pdb
#import warnings
#warnings.filterwarnings("error")

##################################
# Return a new empty BITnet instance

def newNet():

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
               step_num=0)
    
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

    # Check constraints etc
    if np.isscalar(thr_range): thr_range = [thr_range,thr_range]
    if np.isscalar(fire_rate): fire_rate = [fire_rate,fire_rate]
    if np.isscalar(const_inp): const_inp = [const_inp,const_inp]

    net['dim'].append(np.array(dims,int))
    net['x'].append(np.zeros(int(np.prod(dims))))
    net['x_prev'].append(np.zeros(int(np.prod(dims))))
    net['x_calc'].append(np.zeros(int(np.prod(dims))))
    net['x_input'].append(np.zeros(int(np.prod(dims))))
    net['x_fastinp'].append(np.zeros(int(np.prod(dims))))
    net['mean_rate'].append(np.zeros(int(np.prod(dims))))
    net['eta_ip'].append(ip_rate)
    if len(thr_range) == 2:
        net['thr'].append(np.random.uniform(thr_range[0],thr_range[1],
                                            int(np.prod(dims))))
    else:
        net['thr'].append(thr_range) # xxx can't do this?
    if len(fire_rate) == 2:
        net['fire_rate'].append(np.random.uniform(fire_rate[0],fire_rate[1],
                                                  int(np.prod(dims))))
        net['have_rate'].append(any(net['fire_rate'][-1]>0.0))
    else:
        net['fire_rate'].append(fire_rate)
        net['have_rate'].append(fire_rate>0.0)
    if len(const_inp) == 2:
        net['const_inp'].append(np.random.uniform(const_inp[0],const_inp[1],
                                                  int(np.prod(dims))))
    else:
        net['const_inp'].append(const_inp)
    net['nois'].append(nois)
    net['set_spks'].append([])
    net['sspk_idx'].append(0)
    net['spikes'].append([])
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

    # Check constraints etc
    if np.isscalar(W_range): W_range = [W_range,W_range]

    nrow = np.size(net['x'][layer_pre])
    ncol = np.size(net['x'][layer_post])
    if False:
        net['W'].append(np.random.uniform(W_range[0],W_range[1],[nrow,ncol]))
    else:
        Wmn = (W_range[0]+W_range[1])/2.0
        Wsd = (W_range[1]-W_range[0])/6.0
        net['W'].append(np.random.normal(Wmn,Wsd,[nrow,ncol]))
        if Wmn > 0.0:
            net['W'][-1][net['W'][-1]<0.0] = 0.0
        else:
            net['W'][-1][net['W'][-1]>0.0] = 0.0
    setzero = np.random.rand(nrow,ncol) > p
    if layer_pre==layer_post: # no self connections allowed
        setzero = np.logical_or(setzero,np.identity(nrow))
    net['W'][-1][setzero] = 0.0
    net['W_lyr'].append([layer_pre,layer_post])
    net['I'].append(np.zeros(ncol))
    net['eta_stdp'].append(stdp_rate)
    net['is_inhib'].append(W_range[0]<0.0 and W_range[1]<=0.0)
    net['fast_inhib'].append(fast_inhib and net['is_inhib'][-1])
    p_nz = p
    if p_nz==0.0: p_nz = 1.0

    # Normalise the weights (except fast inhib weights)
    if not net['fast_inhib'][-1]:
        nrm = np.linalg.norm(net['W'][-1],ord=1,axis=0)
        nrm[nrm==0.0] = 1.0
        net['W'][-1] = net['W'][-1]/nrm
    
    return len(net['W'])-1

##################################
# Set defined spike times for a neuron layer (ie a neuron population)
#  net:   BITnet instance
#  layer: layer number
#  times: 2-column matrix (col 1 = step num (ordered); col 2 = neuron num to spike)
# NOTE for spike forcing an output layer ensure that: eta_ip=0 and target fire_rate=0
# FOLLOWING training ensure that: forced spikes array is removed, ie: setSpikeTimes(n,l,[])

def setSpikeTimes(net,layer,times):

    net['set_spks'][layer] = times.copy()
    net['sspk_idx'][layer] = 0
    
##################################
# Normalise all the firing rates
#  net: BITnet instance

def norm_rates(net):

    for i,rate in enumerate(net['fire_rate']):
        if rate.any() and net['eta_ip'][i] > 0.0:
            net['thr'][i] = net['thr'][i] + net['eta_ip'][i]*((net['x'][i]>0.0)-rate)
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

def calc_spikes(net):

    # Start with the noise and constant input in the neurons of each layer
    for i,nois in enumerate(net['nois']):
        if nois > 0:
            net['x_input'][i] = np.random.normal(0.0,nois,int(np.prod(net['dim'][i])))
        else:
            net['x_input'][i].fill(0.0)
        net['x_input'][i] += net['const_inp'][i].copy()
        # Find the threshold crossings (overwritten later if needed)
        net['x'][i] = np.clip(net['x_input'][i]-net['thr'][i],a_min=0.0,a_max=0.9)
    # Loop thru layers to insert any predefined spikes
    for i in range(len(net['set_spks'])):
        if len(net['set_spks'][i]):
            net['x'][i].fill(0.0)
            sidx = net['sspk_idx'][i]
            if sidx < len(net['set_spks'][i]): stim = net['set_spks'][i][sidx,0]
            while sidx < len(net['set_spks'][i]) and int(stim) <= net['step_num']:
                net['x'][i][int(net['set_spks'][i][sidx,1])] = np.mod(stim,1)
                sidx += 1
                if sidx < len(net['set_spks'][i]):
                    stim = net['set_spks'][i][sidx,0]
                #else:
                #    net['set_spks'][i] = []
            net['sspk_idx'][i] = sidx

    # Loop thru weight matrices, propagating spikes through.
    # The idea is to process all weight matrices going into a layer (ie the nett input to that layer)
    # then calculate that layer's spikes (threshold crossings), then move to the next group of weight
    # matrices for the next layer. A group is defined as a contiguous set of weight matrices all ending
    # on the same layer. This scheme is designed to propagate spikes rapidly up a feedfoward
    # hierarachy. It won't work for layers with recurrent connections even if they are in the same
    # weight group, since the spikes won't be recurrently p[numnrocessed until the next timestep, so fast
    # inhibition is still needed for that. For feedback connections (ie the same layer being in
    # different weight groups) this code will do a double timestep for those layers (not desirable).
    #ipdb.set_trace()
    for i,W in enumerate(net['W']):
        if not net['fast_inhib'][i]:
            layers = net['W_lyr'][i]

            # Synaptic currents last for 1 timestep
            if layers[0]!=layers[1]:
                net['I'][i] = np.matmul(net['x'][layers[0]],W)
            else:
                net['I'][i] = np.matmul(net['x_prev'][layers[0]],W)
            
            net['x_input'][layers[1]] += net['I'][i]

            # Do spikes if this is the last weight matrix or if the next one has a different post layer
            # or the next one is fast inhib,### UNLESS this is a recurrent layer
            do_spikes = (i==len(net['W'])-1)
            if not do_spikes:
                do_spikes = not(layers[1]==net['W_lyr'][i+1][1]) or net['fast_inhib'][i+1]
                #if do_spikes:
                #    do_spikes = layers[0]!=layers[1]
            if do_spikes:
                j = layers[1]

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
            n_idx = np.nonzero(net['x'][i])[0]
            net['spikes'][i].extend([net['step_num']+1-net['x'][i][n],n]
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
            shape = np.shape(W)

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
                pre  = np.tile(np.reshape(mpre, [shape[0],1]),[1,shape[1]])
                post = np.tile(np.reshape(xdiff,[1,shape[1]]),[shape[0],1])

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

                # Pre and Post spikes tiled across and down for all synapses
                #if net['have_rate'][layers[0]]:
                #    # Modulate learning rate by firing rate (low firing rate = high learning rate)
                #    mpre = net['x'][layers[0]] #/net['fire_rate'][layers[0]]
                #else:
                #    mpre = net['x'][layers[0]]
                #pre = np.tile(np.reshape(mpre,[shape[0],1]),[1,shape[1]])
                pre = np.tile(np.reshape(net['x'][layers[0]],[shape[0],1]),[1,shape[1]])
                if net['have_rate'][layers[1]]:
                    # Modulate learning rate by firing rate (low firing rate = high learning rate)
                    mpost = net['x'][layers[1]] #/net['fire_rate'][layers[1]]
                else:
                    mpost = net['x'][layers[1]]
                post = np.tile(np.reshape(mpost,[1,shape[1]]),[shape[0],1])

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

def runSim(net,n_steps):

    # Loop
    for step in range(n_steps):

        # Inc step count
        net['step_num'] += 1

        # Propagate spikes from pre to post neurons
        calc_spikes(net)
        
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
        if any(sp):
            x=[]; y=[]
            for s in sp:
                if s[0] > cutoff:
                    x.extend([s[0]])
                    y.extend([s[1]+n_tot])
            plt.plot(x,y,'.',ms=1)
            n_tot += np.size(net['x'][i])
    
##################################
# Plot recorded spikes in new figure
#  net: BITnet instance

def plotSpikes(net,cutoff):

    plt.figure()
    subplotSpikes(net,cutoff)
    plt.show(block=False)
    
##################################
