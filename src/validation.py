#MIT License

#Copyright (c) 2023 Adam Hines, Michael Milford, Tobias Fischer

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
import pickle
import timeit
import torch
import math

import utils as ut
import numpy as np
import blitnet as bn
import matplotlib.pyplot as plt

from metrics import createPR

# used to validate the network training, reject network training if threshold not met
def validate(model): 
        
    # unpickle the network
    model.logger.info('Unpickling the network')
    with open(model.training_out+'net.pkl', 'rb') as f:
        net = pickle.load(f)
    
    model.logger.info('Validating network training')
    # set input spikes for training data from one location
    net['set_spks'][0] = ut.setSpikeRates(
                        model.imgs['training'][0:model.number_training_images],
                        model.ids['training'][0:model.number_training_images],
                        model.device,
                        [model.imWidth,model.imHeight],
                        True,
                        model.number_training_images,
                        model.number_modules,
                        model.intensity,
                        model.location_repeat)
    
    # correct place matches variable
    numcorrect = 0
    
    if model.validation:
        outconcat = []
    
    # run the test network on training data to evaluate network performance
    start = timeit.default_timer()
    for t in range(model.number_training_images):
        tonump = np.array([])
        bn.testSim(net, device=model.device)
        # output the index of highest amplitude spike
        tonump = np.append(tonump,
                            np.reshape(net['x'][-1].cpu().numpy(),
                            [1,1,int(model.number_training_images)]))
        nidx = np.argmax(tonump)
        
        if model.validation:
            outconcat.append(tonump.tolist())
        
        gt_ind = model.filteredNames.index(model.filteredNames[t])
        
        # adjust number of correct matches if GT matches peak output
        if gt_ind == nidx:
            numcorrect += 1
            
    end = timeit.default_timer()
    
    # network must match >75% of training places to be successful
    p100r = (numcorrect/model.number_training_images)*100
    testFlag = (p100r>75)
    if testFlag: # network training was successful
        
        model.logger.info('')
        model.logger.info('Network training successful!')
        model.logger.info('')
        
        model.logger.info('Performance details:')
        model.logger.info("-------------------------------------------")
        model.logger.info('P@100R: '+str(p100r)+
              '%  |  Query frequency: '+
              str(round(model.number_training_images/(end-start),2))+'Hz')
        model.logger.info('')
        
        # create output folder (if it does not already exist)
        if not os.path.isdir(model.training_out):
            os.mkdir(model.training_out)
        if not os.path.isdir(model.training_out+'images/'):
            os.mkdir(model.training_out+'images/')
        if not os.path.isdir(model.training_out+'images/training/'):
            os.mkdir(model.training_out+'images/training/')
        
        # plot the similarity matrix for network validation
        if model.validation:
            outconcat = np.array(outconcat)
            concatReshape = np.reshape(outconcat,
                                        (model.number_training_images,
                                        model.number_training_images))
            folderName = model.output_folder+'/similarity/'
            os.mkdir(folderName)

            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            fig.suptitle('Training similarity', fontsize = 18)
            cmap = plt.cm.gist_yarg
            plot_name = "Similarity: network training validation"
            ut.plot_similarity(concatReshape, 
                               plot_name, 
                               cmap, 
                               ax=axes)
            
        # Reset network details
        net['sspk_idx'] = [0,0,0]
        net['step_num'] = 0
        net['spikes'] = [[],[],[]]
        net['x'] = [[],[],[]]
        net['set_spks'][0] = 0
        
        # plot the weight matrices
        cmap = plt.cm.magma
        cmap_reverse = plt.cm.magma_r
        
        model.logger.info('Plotting weight matrices')
        # make weights folder
        weights_folder = model.output_folder+'/weights/'
        os.mkdir(weights_folder)
        
        # get the maximum weight value for each set
        IF_Exc = torch.max(net['W'][0]).cpu().numpy()
        IF_Inh = torch.min(net['W'][1]).cpu().numpy()
        FO_Exc = torch.max(net['W'][2]).cpu().numpy()
        FO_Inh = torch.min(net['W'][3]).cpu().numpy()
        
        # find highest divisors of weight matrices for plotting
        def closestDivisors(n):
            factor1 = round(math.sqrt(n))
            while n%factor1 > 0: factor1 -= 1
            return factor1, n//factor1
        
        factor1IF, factor2IF = closestDivisors(len(net['W'][0][0][0])*len(net['W'][0][0])*model.number_modules)
        factor1FO, factor2FO = closestDivisors(len(net['W'][2][0][0])*len(net['W'][2][0])*model.number_modules)
        
        # initial weights
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Initial weights', fontsize = 18)
        
        ut.plot_weights(W=model.init_weights[0], 
                        name='I->F Excitatory Weights', 
                        cmap=cmap, 
                        vmax=IF_Exc, 
                        dims=[factor1IF, factor2IF, int((factor1IF*factor2IF)/model.number_modules)], 
                        ax=axes[0, 0])

        ut.plot_weights(W=model.init_weights[1], 
                        name='I->F Inhibitory Weights', 
                        cmap=cmap_reverse, 
                        vmax=IF_Inh, 
                        dims=[factor1IF, factor2IF, int((factor1IF*factor2IF)/model.number_modules)], 
                        ax=axes[0, 1])

        ut.plot_weights(W=model.init_weights[2], 
                        name='F->O Excitatory Weights', 
                        cmap=cmap, 
                        vmax=FO_Exc, 
                        dims=[factor1FO, factor2FO, int((factor1FO*factor2FO)/model.number_modules)],
                          ax=axes[1, 0])

        ut.plot_weights(W=model.init_weights[3], 
                        name='F->O Inhibitory Weights', 
                        cmap=cmap_reverse, 
                        vmax=FO_Inh, 
                        dims=[factor1FO, factor2FO, int((factor1FO*factor2FO)/model.number_modules)], 
                        ax=axes[1, 1])


        plt.tight_layout()
        plt.show()

        fig.savefig(weights_folder + 'Combined_Weights.pdf', dpi=600)
        
        # calculated weights
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Calculated weights', fontsize = 18)

        ut.plot_weights(W=net['W'][0], 
                        name='I->F Excitatory Weights', 
                        cmap=cmap,
                        vmax=IF_Exc, 
                        dims=[factor1IF, factor2IF, int((factor1IF*factor2IF)/model.number_modules)], 
                        ax=axes[0, 0])

        ut.plot_weights(W=net['W'][1], 
                        name='I->F Inhibitory Weights', 
                        cmap=cmap_reverse, 
                        vmax=IF_Inh, 
                        dims=[factor1IF, factor2IF, int((factor1IF*factor2IF)/model.number_modules)], 
                        ax=axes[0, 1])

        ut.plot_weights(W=net['W'][2], 
                        name='F->O Excitatory Weights', 
                        cmap=cmap, 
                        vmax=FO_Exc, 
                        dims=[factor1FO, factor2FO, int((factor1FO*factor2FO)/model.number_modules)], 
                        ax=axes[1, 0])

        ut.plot_weights(W=net['W'][3], 
                        name='F->O Inhibitory Weights', 
                        cmap=cmap_reverse, 
                        vmax=FO_Inh, 
                        dims=[factor1FO, factor2FO, int((factor1FO*factor2FO)/model.number_modules)], 
                        ax=axes[1, 1])


        plt.tight_layout()
        plt.show()

        fig.savefig(weights_folder + 'Combined_Weights.pdf', dpi=600)

        
    else:
        # log unsuccessful training details
        model.logger.info('')
        model.logger.info('Network training unsuccessful.')
        model.logger.info('')
        model.logger.info('Performance details:')
        model.logger.info("-------------------------------------------")
        model.logger.info('P@100R: '+str(p100r)+
              '%  |  Query frequency: '+
              str(round(model.number_training_images/(end-start),2))+'Hz')
        model.logger.info('')
    return testFlag

# run PR and recall at N
def match_metrics(numpconc, output_folder, number_testing_images, number_training_images, logger):
    
    numpconc = np.array(numpconc).T

    # make the output folder
    folderName = output_folder+'/similarity/'
    if not os.path.isdir(folderName):
        os.mkdir(folderName)

    # reshape similarity matrix
    sim_mat = np.reshape(numpconc,(number_testing_images, number_training_images))

    # generate the ground truth matrix
    GT = np.zeros((number_testing_images, number_training_images), dtype=int)
    for n in range(len(GT)):
        GT[n,n] = 1

    # create the main figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Network metrics', fontsize = 18)
    cmap = plt.cm.tab20c
        
    # plot the similarity matrices
    ut.plot_similarity(sim_mat, 'VPRTempo similarity', cmap, ax=axes[0,0])
    ut.plot_similarity(GT, 'Ground truth', cmap, ax=axes[0,1])

    # get the P & R 
    P, R = createPR(sim_mat, GT, GT, matching="single")
    for n, ndx in enumerate(P):
        P[n] = round(ndx,2)
        R[n] = round(R[n],2)
        
    logger.info('Precision values: '+str(P))
    logger.info('Recall values: '+str(R))

    # plot the PR curve
    ut.plot_PR(P, R, 'Precision-recall curve',ax=axes[1,0])

    # calculate the recall at N
    N_vals = [1, 5, 10, 15, 20, 25]
    recallN = ut.recallAtN(sim_mat, GT, GT, N_vals)

    # plot the recall at N
    ut.plot_recallN(recallN, N_vals, 'Recall@N',ax=axes[1,1])

    # show the full plot
    plt.tight_layout()
    plt.show()

    logger.info('')
    for n, ndx in enumerate(recallN):
        logger.info('Recall at N='+str(N_vals[n])+': '+str(round(ndx,2)))
