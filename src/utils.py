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

# Get the 2D patches or the patch normalization

'''
Imports
'''
import cv2
import os
import math
import torch
import pickle
import timeit

import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import blitnet as bn

from metrics import recallAtK, createPR
from timeit import default_timer


def get_patches2D(patch_size,image_pad):
    
    if patch_size[0] % 2 == 0: 
        nrows = image_pad.shape[0] - patch_size[0] + 2
        ncols = image_pad.shape[1] - patch_size[1] + 2
    else:
        nrows = image_pad.shape[0] - patch_size[0] + 1
        ncols = image_pad.shape[1] - patch_size[1] + 1
    patches = np.lib.stride_tricks.as_strided(image_pad , patch_size + (nrows, ncols), 
          image_pad.strides + image_pad.strides).reshape(patch_size[0]*patch_size[1],-1)
    
    return patches

# Run patch normalization on imported RGB images
def patch_normalise_pad(img,patches):
    
    patch_size = (patches, patches)
    patch_half_size = [int((p-1)/2) for p in patch_size ]

    image_pad = np.pad(np.float64(img), patch_half_size, 'constant', 
                                                   constant_values=np.nan)

    nrows = img.shape[0]
    ncols = img.shape[1]
    patches = get_patches2D(patch_size,image_pad)
    mus = np.nanmean(patches, 0)
    stds = np.nanstd(patches, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        im_norm = (img - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)

    im_norm[np.isnan(im_norm)] = 0.0
    im_norm[im_norm < -1.0] = -1.0
    im_norm[im_norm > 1.0] = 1.0
    
    return im_norm

# Process the loaded images - resize, normalize color, & patch normalize
def processImage(img,dims,patches):
    # gamma correct images
    mid = 0.5
    mean = np.mean(img)
    gamma = math.log(mid*255)/math.log(mean)
    img = np.power(img,gamma).clip(0,255).astype(np.uint8)
    
    # resize image to 28x28 and patch normalize        
    img = cv2.resize(img,(dims[0], dims[1]))
    im_norm = patch_normalise_pad(img,patches) 
    img = np.uint8(255.0 * (1 + im_norm) / 2.0)

    return img

# Image loader function - runs all image import functions
def loadImages(test_true,train_paths,img_names,dims,patches,testPath,testLoc):
    
    # get torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    imgs = []
    ids = []

    if test_true:
        train_paths = [testPath+testLoc+'/']
    
    if isinstance(train_paths,list):
        for paths in train_paths:
            for m in img_names:
                fullpath = paths+m
                # read and convert image from BGR to RGB 
                img = cv2.imread(fullpath)[:,:,::-1]

                # convert image
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                imgProc = processImage(img,dims,patches)
                imgs.append(torch.tensor(imgProc,device=device))
                ids.append(m)
    else:
        for m in img_names:
            fullpath = train_paths+m
            # read and convert image from BGR to RGB 
            img = cv2.imread(fullpath)[:,:,::-1]
            # convert image
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            imgProc = processImage(img,dims,patches)
            imgs.append(torch.tensor(imgProc,device=device))
            ids.append(m)
    
    return imgs, ids
             
# sets the spike rates from imported images - convert pixel range [0,255] to [0,1]
# spike rates are set as a 3D tensor with dimensions (m x r x s)
# m = module
# r = location repeat
# s = spikes for number of training images
def setSpikeRates(data,ids,device,dims,test_true,numImgs,numMods,intensity,locRep):   
    
     # output tensor dimensions
     n_input = dims[0] * dims[1]
     
     # Set the spike rates based on the number of example training images
     
     # loop through data and append spike rates for each image
     # organise into 3D tensor based on number of expert modules
     if test_true: # if loading testing data (repeat input across modules)
         init_rates = torch.empty(0,device=device) 
         for m in range(numImgs):

             init_rates = torch.cat((init_rates, torch.reshape(data[m]/intensity,(n_input,)))) 
             
         init_rates = torch.unsqueeze(init_rates,0)    
         # define the initial spike rates
         for o in range(numMods):
             if o == 0:
                 spike_rates = torch.unsqueeze(init_rates,0)
             else:
                 spike_rates = torch.concat((spike_rates,torch.unsqueeze(init_rates,0)),0)

                        
     else: # if loading training data, have separate inputs across modules
        for o in range(numMods):
            start = []
            end = []
            for j in range(locRep):
                mod = j * numImgs
                start.append(int(numImgs/numMods)*(o) + mod)
                end.append(int(numImgs/numMods)*(o + 1) + mod)

            # define the initial spike rates for location repeats
            for m in range(locRep):
                rates = torch.empty(0,device=device)
                for jdx, j in enumerate(range(start[m],end[m])):    
                    rates = torch.cat((rates,
                               torch.reshape(data[j]/intensity,(n_input,))),0)  
                if m == 0:
                    init_rates = torch.unsqueeze(rates,0)
                else:
                    init_rates = torch.cat((init_rates,torch.unsqueeze(rates,0)),-1)
            
            # output spike rates into modules
            if o == 0: # append the location repeat initial spikes to a new module
                spike_rates = torch.unsqueeze(init_rates,0)
            else:
                spike_rates = torch.concat((spike_rates,torch.unsqueeze(init_rates,0)),0)
     
     return spike_rates

# plot similarity matrices
def plot_similarity(mat,name,outfold,cmap):
    
    fig = plt.figure()
    plt.matshow(mat,fig, cmap=cmap)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.colorbar(label="Spike amplitude")
    fig.suptitle(name,fontsize = 12)
    plt.xlabel("Query",fontsize = 12)
    plt.ylabel("Database",fontsize = 12)
    plt.show()
    fig.savefig(outfold+name+'.pdf', dpi=600)
    

# plot weight matrices
def plot_weights(W, name, cmap, vmax, dims, ax=None):
    newx = dims[0]
    newy = dims[1]
    
    # loop through expert modules and output weights
    init_weight = np.array([])
    for n in range(len(W[:,0,0])):
        init_weight = np.append(init_weight, np.reshape(W[n,:,:].cpu().numpy(), (dims[2],)))
    
    # reshape the weight matrices
    reshape_weight = np.reshape(init_weight, (newx, newy))
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    cax = ax.matshow(reshape_weight, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(cax, ax=ax, label="Weight strength")
    
    ax.set_title(name, fontsize = 12)
    ax.set_xlabel("x-weights", fontsize = 12)
    ax.set_ylabel("y-weights", fontsize = 12)
    
# run recallAtK() function from VPR Tutorial
def recallAtN(S_in, GThard, GTsoft, N):
    
    # run recall at N over each value of N
    recall_list = []
    for n in N:
        recall_list.append(recallAtK(S_in, GThard, GTsoft, K=n))
        
    return recall_list
      
def sad(fullTrainPaths, filteredNames, imWidth, imHeight, num_patches, testPath, 
        test_location, imgs, ids, number_testing_images, number_training_images,
        validation):

    print('')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Setting up Sum of Absolute Differences (SAD) calculations')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('')

    sadcorrect = 0
    # load the training images
    imgs['training'], ids['training'] = ut.loadImages(False, 
                                                      fullTrainPaths, 
                                                      filteredNames, 
                                                      [imWidth,imHeight], 
                                                      num_patches, 
                                                      testPath, 
                                                      test_location)
    del imgs['training'][number_training_images:len(imgs['training'])]
    
    # create database tensor
    for ndx, n in enumerate(imgs['training']):
        if ndx == 0:
            db = torch.unsqueeze(n,0)
        else:
            db = torch.cat((db, torch.unsqueeze(n,0)), 0)

    def calc_sad(query, database, const):
        SAD = torch.sum(torch.abs((database * const) - (query * const)), (1,2), keepdim=True)
        for n in range(2):
            SAD = torch.squeeze(SAD,-1)
        return SAD

    # calculate SAD for each image to database and count correct number
    imgred = 1/(imWidth * imHeight)
    sad_concat = []

    print('Running SAD')
    correctidx = []
    incorrectidx = []

    start = default_timer()
    for n, q in enumerate(imgs['testing']):
        pixels = torch.empty([])

        # create 3D tensor of query images
        for o in range(number_testing_images):
            if o == 0:
                pixels = torch.unsqueeze(q,0)
            else:
                pixels = torch.cat((pixels,torch.unsqueeze(q,0)),0)
        
        sad_score = calc_sad(pixels, db, imgred)
        best_match = np.argmin(sad_score.cpu().numpy())

        if n == best_match:
            sadcorrect += 1
            correctidx.append(n)
        else:
            incorrectidx.append(n)

        if validation:
            sad_concat.append(sad_score.cpu().numpy())

    end = default_timer()   

    p100r_local = round((sadcorrect/number_testing_images)*100,2)
    print('')
    print('Sum of absolute differences P@1: '+ str(p100r_local) + '%')
    print('Sum of absolute differences queried at ' + str(round(number_testing_images/(end-start),2)) + 'Hz')
    
    GT = np.zeros((number_testing_images,number_training_images), dtype=int)
    for n in range(len(GT)):
        GT[n,n] = 1
    sad_concat = (1-np.reshape(np.array(sad_concat),(number_training_images,number_testing_images)))
    P,R  = createPR(sad_concat,GT,GT,matching="single")
    for n, ndx in enumerate(P):
        P[n] = round(ndx,2)
        R[n] = round(R[n],2)

    # make the PR curve
    fig = plt.figure()
    plt.plot(R,P)
    fig.suptitle("Precision Recall curve",fontsize = 12)
    plt.xlabel("Recall",fontsize = 12)
    plt.ylabel("Precision",fontsize = 12)
    plt.show()
    
    # calculate the recall at N
    N_vals = [1,5,10,15,20,25]
    recallN = ut.recallAtN(sad_concat, GT, GT, N_vals)
    
    return P,R,recallN,N_vals

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
            plot_name = "Similarity: network training validation"
            ut.plot_similarity(concatReshape, plot_name,
                                folderName,
                                plt.cm.gist_yarg)
            
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
