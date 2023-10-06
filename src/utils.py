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

import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import blitnet as bn

from metrics import recallAtK, createPR
from timeit import default_timer
from os import path

def dummy_bmm(device):
    # Create some dummy tensors on CUDA
    dummy_a = torch.randn(10, 10, device=device)
    dummy_b = torch.randn(10, 10, device=device)
    
    # Perform a dummy bmm operation
    torch.bmm(dummy_a.unsqueeze(0), dummy_b.unsqueeze(0))

def sad(self):

    print('')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Setting up Sum of Absolute Differences (SAD) calculations')    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('')

    sadcorrect = 0

    # load the training images
    self.location_repeat = 1 # switch to only do SAD on one dataset traversal
    self.fullTrainPaths = self.fullTrainPaths[1]
    self.test_true = False # testing images preloaded, load the training ones
    # load the training images
    self.imgs['training'], self.ids['training'] = ut.loadImages(self.test_true,
                                        self.fullTrainPaths,
                                        self.filteredNames,
                                        [self.imWidth,self.imHeight],
                                        self.num_patches,
                                        self.testPath,
                                        self.test_location)

    # create database tensor
    for ndx, n in enumerate(self.imgs['training']):
        if ndx == 0:
            db = torch.unsqueeze(n,0)
        else:
            db = torch.concat((db,torch.unsqueeze(n,0)),0)

    def calc_sad(query, database, const):

        SAD = torch.sum(torch.abs(torch.sub((database * const), (query * const))),
                        (1,2),keepdim=True)
        for n in range(2):
            SAD = torch.squeeze(SAD,-1)
        return SAD

    # calculate SAD for each image to database and count correct number
    imgred = 1/(self.imWidth*self.imHeight)
    sad_concat = []
    print('Running SAD')

    start = timeit.default_timer()
    for n, q in enumerate(self.imgs['testing']):
        pixels = torch.empty([])
         # create 3D tensor of query images
        for o in range(self.number_testing_images):
            if o == 0:
                pixels = torch.unsqueeze(q,0)
            else:
                pixels = torch.concat((pixels,torch.unsqueeze(q,0)),0)

        sad_score = calc_sad(pixels, db, imgred)

        best_match = np.argmin(sad_score.cpu().numpy())
        if n == best_match:
            sadcorrect += 1

    end = timeit.default_timer()   
    p100r = round((sadcorrect/self.number_testing_images)*100,2)
    print('')
    print('Sum of absolute differences P@1: '+
          str(p100r)+'%')
    print('Sum of absolute differences queried at '
          +str(round(self.number_testing_images/(end-start),2))+'Hz')


# plot similarity matrices
def plot_similarity(mat, name, cmap, ax=None, dpi=600):
    
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi,figsize=(8, 6))
    else:
        fig = ax.get_figure()

    cax = ax.matshow(mat, cmap=cmap, aspect='equal')
    fig.colorbar(cax, ax=ax, label="Spike amplitude")
    ax.set_title(name,fontsize = 12)
    ax.set_xlabel("Query",fontsize = 12)
    ax.set_ylabel("Database",fontsize = 12)
    
    

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

    # plot the weight matrix to specified subplot
    cax = ax.matshow(reshape_weight, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(cax, ax=ax, label="Weight strength",shrink=0.5)
    
    # set figure titles and labels
    ax.set_title(name, fontsize = 12)
    ax.set_xlabel("x-weights", fontsize = 12)
    ax.set_ylabel("y-weights", fontsize = 12)

# plot PR curves
def plot_PR(P, R, name, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(R, P)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)

# plot the recall@N
def plot_recallN(recallN, N_vals, name, ax=None):
    
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
    
        ax.plot(N_vals, recallN)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("N", fontsize=12)
        ax.set_ylabel("Recall", fontsize=12)
    
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

# clear the contents of the weights folder if retraining with same settings
def clear_weights(training_out):
    if os.path.isfile(training_out + 'net.pkl'):
        os.remove(training_out+'net.pkl')
    if os.path.isfile(training_out + 'GT_imgnames.pkl'):
        os.remove(training_out+'GT_imgnames.pkl')
    if not os.path.isdir(training_out):
        os.mkdir(training_out)