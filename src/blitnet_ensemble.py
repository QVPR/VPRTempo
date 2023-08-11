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


##################################
# Propagate spikes thru the network
#  net: SORN instance

def calc_spikes(net):

    # Start with the noise and constant input in the neurons of each layer
    for i,nois in enumerate(net['nois']):
        if nois > 0:
            net['x_input'][i] = np.random.normal(0.0,nois,int(np.prod(net['dim'][i])))
        else:
            if net['x_input'][i].dim() < 2:
                net['x_input'][i] = torch.unsqueeze(torch.full_like(net['x_input'][i],0.0),-1)
            elif net['x_input'][i].dim() > 2:
                net['x_input'][i] = torch.squeeze(torch.full_like(net['x_input'][i],0.0),-1)
            else:
                net['x_input'][i] = torch.full_like(net['x_input'][i],0.0)
            
            if net['x_input'][i].size(dim=1) > net['n_ensemble']:
                temptens = torch.hsplit(net['x_input'][i],2)
                net['x_input'][i] = temptens[0]
            elif net['x_input'][i].size(dim=1) < net['n_ensemble']:
                tempinput = torch.clone(net['x_input'][i])
                for n in range(len(net['const_inp'][0][-1])-1):
                    net['x_input'][i] = torch.concat((net['x_input'][i],tempinput),-1)

        net['x_input'][i] += net['const_inp'][i].detach().clone()
        # Find the threshold crossings (overwritten later if needed)
        net['x'][i] = torch.clamp((net['x_input'][i]-net['thr'][i]),0.0,0.9)

    # Loop thru layers to insert any predefined spikes
    for i in range(len(net['set_spks'])):
        if len(net['set_spks'][i]):
            net['x'][i] = torch.full_like(net['x'][i],0.0)
            sidx = net['sspk_idx'][i]
            if sidx < len(net['set_spks'][i]): stim = net['set_spks'][i][sidx,0,:]
            while sidx < len(net['set_spks'][i]) and int(stim[0]) <= net['step_num']:
                net['x'][i][int(net['set_spks'][i][sidx,1][0])] = torch.fmod(stim,1)
                sidx += 1
                if sidx < len(net['set_spks'][i]):
                    stim = net['set_spks'][i][sidx,0]
                #else:
                #    net['set_spks'][i] = []
            net['sspk_idx'][i] = sidx
        pause=1

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
    def batch_mm(matrix, vector_batch):
        batch_size = vector_batch.shape[0]
        # Stack the vector batch into columns. (b, n, 1) -> (n, b)
        vectors = vector_batch.transpose(0, 1).reshape(-1, batch_size)
    
        # A matrix-matrix product is a batched matrix-vector product of the columns.
        # And then reverse the reshaping. (m, b) -> (b, m, 1)
        return matrix.mm(vectors).transpose(1, 0).reshape(batch_size, -1, 1)
    
    for i,W in enumerate(net['W']):
        if not net['fast_inhib'][i]:
            layers = net['W_lyr'][i]

            # Synaptic currents last for 1 timestep
            if layers[0]!=layers[1]:
                net['I'][i] = torch.matmul(net['x'][layers[0]],W)
                #net['I'][i] = torch.einsum('bi,bji->ji',net['x'][layers[0]],W)
            else:
                net['I'][i] = torch.einsum('bi,bji->ji',net['x_prev'][layers[0]],W)
            
            net['x_input'][layers[1]] += net['I'][i]
            a = torch.squeeze(net['x_input'][layers[1]],-1).numpy()
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
            outspk = (net['x'][i]).detach().cpu().numpy()
            if i == 2:
                outspk[outspk<0.05] = 0
            n_idx = np.nonzero(outspk)
            net['spikes'][i].extend([net['step_num']+net['x'][i][n].detach().cpu().numpy(),n]
                                        for n in n_idx)
            
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