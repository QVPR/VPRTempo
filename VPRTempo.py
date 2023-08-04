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

'''
Imports
'''
import cv2
import pickle
import os
import torch
import gc
import math
import timeit
import sys
sys.path.append('./src')
sys.path.append('./weights')
sys.path.append('./settings')
sys.path.append('./output')

import numpy as np
import blitnet_open as blitnet
import blitnet_ensemble as ensemble
import validation as validate
import matplotlib.pyplot as plt

from os import path
from alive_progress import alive_bar
from metrics import createPR
from timeit import default_timer as timer


'''
Spiking network model class
'''
class snn_model():
    def __init__(self):
        super().__init__()
        
        '''
        USER SETTINGS
        '''
        self.trainingPath = '/home/adam/data/hpc/' # training datapath
        self.testPath = '/home/adam/data/testing_data/' # testing datapath
        self.number_training_images = 100 # alter number of training images
        self.number_ensembles = 30 # number of ensemble networks
        self.ensemble_max = self.number_training_images*10 # maximum number of images per ensemble_net
        self.location_repeat = 2 # Number of training locations that are the same
        self.locations = ["spring","fall"] # which datasets are used in the training
        self.test_location = "summer"
        
        # Image and patch normalization settings
        self.imWidth = 28 # image width for patch norm
        self.imHeight = 28 # image height for patch norm
        self.num_patches = 7 # number of patches
        self.intensity = 255 # divide pixel values to get spikes in range [0,1]
        
        # Network and training settings
        self.input_layer = (self.imWidth*self.imHeight) # number of input layer neurons
        self.feature_layer = int(self.input_layer*2) # number of feature layer neurons
        self.output_layer = self.number_training_images # number of output layer neurons (match training images)
        self.train_img = self.output_layer # number of training images
        self.epoch = 4 # number of training iterations
        self.test_t = self.output_layer # number of testing time points
        self.cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # saliency calculating on cpu or gpu
        self.T = int(self.number_training_images*self.epoch) # number of training steps
        self.annl_pow = 2 # learning rate anneal power
        self.filter = 8 # filter images every 8 seconds (equivelant to 8 images)
        
        # Select training images from list
        with open('./nordland_imageNames.txt') as file:
            self.imageNames = [line.rstrip() for line in file]
            
        # Filter the loading images based on self.filter
        self.filteredNames = []
        for n in range(0,len(self.imageNames),self.filter):
            self.filteredNames.append(self.imageNames[n])
        del self.filteredNames[self.number_training_images*self.number_ensembles:len(self.filteredNames)]
        
        # Get the full training and testing data paths    
        self.fullTrainPaths = []
        for n in self.locations:
                self.fullTrainPaths.append(self.trainingPath+n+'/')
        
        # Hyperparamters
        self.theta_max = 0.25 # maximum threshold value
        self.n_init = 0.01 # initial learning rate value
        self.n_itp = 0.25 # initial intrinsic threshold plasticity rate[[0.9999]]
        self.f_rate = [0.0012,0.2]# firing rate range
        self.p_exc = 0.025 # probability of excitatory connection
        self.p_inh = 0.125 # probability of inhibitory connection
        self.c= 0.125 # constant input 
        
        # Test settings 
        self.test_true = False # leave default to False
        self.validation = True # test this network against other methods, default True
        
        # Print network details
        print('////////////')
        print('VPRTempo - Temporally Encoded Visual Place Recognition v0.1')
        print('Queensland University of Technology, Centre for Robotics')
        print('\\\\\\\\\\\\\\\\\\\\\\\\')
        print('Theta: '+str(self.theta_max))
        print('Initial learning rate: '+str(self.n_init))
        print('ITP Learning: '+str(self.n_itp))
        print('Firing rate: '+str(self.f_rate[0]) +' to '+ str(self.f_rate[1]))
        print('Excitatory p: '+str(self.p_exc))
        print('Inhibitory p: '+str(self.p_inh))
        print('Constant input '+str(self.c))
        print('CUDA available: '+str(torch.cuda.is_available()))
        if torch.cuda.is_available() == True:
            current_device = torch.cuda.current_device()
            print('Current device is: '+str(torch.cuda.get_device_name(current_device)))
        else:
            print('Current device is: CPU')
 
        # Network weights name
        self.training_out = './weights/'+str(self.input_layer)+'i'+\
                                            str(self.feature_layer)+\
                                        'f'+str(self.output_layer)+\
                                            'o'+str(self.epoch)+'/'
  
    # Get the 2D patches or the patch normalization
    def get_patches2D(self):
        
        if self.patch_size[0] % 2 == 0: 
            nrows = self.image_pad.shape[0] - self.patch_size[0] + 2
            ncols = self.image_pad.shape[1] - self.patch_size[1] + 2
        else:
            nrows = self.image_pad.shape[0] - self.patch_size[0] + 1
            ncols = self.image_pad.shape[1] - self.patch_size[1] + 1
        self.patches = np.lib.stride_tricks.as_strided(self.image_pad , self.patch_size + (nrows, ncols), 
          self.image_pad.strides + self.image_pad.strides).reshape(self.patch_size[0]*self.patch_size[1],-1)
    
    # Run patch normalization on imported RGB images
    def patch_normalise_pad(self):
        
        self.patch_size = (self.num_patches, self.num_patches)
        patch_half_size = [int((p-1)/2) for p in self.patch_size ]
    
        self.image_pad = np.pad(np.float64(self.img), patch_half_size, 'constant', 
                                                       constant_values=np.nan)
    
        nrows = self.img.shape[0]
        ncols = self.img.shape[1]
        self.get_patches2D()
        mus = np.nanmean(self.patches, 0)
        stds = np.nanstd(self.patches, 0)
    
        with np.errstate(divide='ignore', invalid='ignore'):
            self.im_norm = (self.img - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)
    
        self.im_norm[np.isnan(self.im_norm)] = 0.0
        self.im_norm[self.im_norm < -1.0] = -1.0
        self.im_norm[self.im_norm > 1.0] = 1.0
    
    # Process the loaded images - resize, normalize color, & patch normalize
    def processImage(self):
        # gamma correct images
        mid = 0.5
        mean = np.mean(self.img)
        gamma = math.log(mid*255)/math.log(mean)
        self.img = np.power(self.img,gamma).clip(0,255).astype(np.uint8)
        
        # resize image to 28x28 and patch normalize        
        self.img = cv2.resize(self.img,(self.imWidth, self.imHeight))
        self.patch_normalise_pad() 
        self.img = np.uint8(255.0 * (1 + self.im_norm) / 2.0)

        
    # Image loader function - runs all image import functions
    def loadImages(self):

        # Create dictionary of images
        self.imgs = {'training':[],'testing':[]}
        self.ids = {'training':[],'testing':[]}
        
        if self.test_true:
            self.fullTrainPaths = [self.testPath+self.test_location+'/']
        
        for paths in self.fullTrainPaths:
            self.dataPath = paths
            if self.test_location in self.dataPath:
                dictEntry = 'testing'
            else:
                dictEntry = 'training'
            for m in self.loadNames:
                self.fullpath = self.dataPath+m
                # read and convert image from BGR to RGB 
                self.img = cv2.imread(self.fullpath)[:,:,::-1]
                # convert image
                self.img = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
                self.processImage()
                self.imgs[dictEntry].append(self.img)
                self.ids[dictEntry].append(m)
                 
    
    def setSpikeRates(self, images, ids):     
         data = {'x': np.array(images), 'y': np.array(ids), 
                             'rows': self.imWidth, 'cols': self.imHeight}
         num_testing_imgs = data['x'].shape[0]
         self.num_examples = num_testing_imgs
         n_input = self.imWidth * self.imHeight 
         
         # Set the spike rates based on the number of example training images
         self.spike_rates = []
         self.init_rates = []
         for jdx, j in enumerate(range(int(self.num_examples))):    
             self.init_rates.append((data['x'][j%num_testing_imgs,:,:].reshape((n_input))/self.intensity))  
         for n in range(int(self.epoch)): 
             self.spike_rates.extend(self.init_rates)
         if not self.test_true:
             self.spike_rates.extend(self.spike_rates)
             
    def checkTrainTest(self):
        # Check if pre-trained network exists, prompt if retrain or run
        if path.isdir(self.training_out) and len(os.listdir(self.training_out)) == self.number_ensembles or os.path.isfile(self.training_out + 'ensemble_net1.pkl'):
            retrain = input("A network with these parameters exists, re-train ensembles? (y/n):\n")
        else:
            retrain = 'y'
        return retrain
    
    '''
    Run the training network
    '''
    def train(self):

        # check if training required
        check = self.checkTrainTest()
        
        if check == 'y':
              
            if os.path.isfile(self.training_out + 'ensemble_net.pkl'):
                os.remove(self.training_out+'ensemble_net.pkl')
            
            def train_start():  
                
                self.copy_max = self.ensemble_max
                self.ensemble_netNum = 1
                breakUp = int(self.ensemble_max/self.number_training_images)
                
                for ensemble_num, ndx in enumerate(range(self.number_ensembles)):
                    start_range = self.number_training_images * ndx
                    end_range = self.number_training_images * (ndx+1)
                    
                    self.loadNames = self.filteredNames[start_range:end_range]
                    self.loadImages()
                    self.setSpikeRates(self.imgs['training'],self.ids['training'])
                    # create new network
                    net = blitnet.newNet()
                    iLayer = blitnet.addLayer(net,[self.input_layer,1],0.0,0.0,0.0,0.0,0.0,
                                                                         False)   
                    fLayer = blitnet.addLayer(net,[self.feature_layer,1],[0,self.theta_max],
                                              [self.f_rate[0],self.f_rate[1]],self.n_itp,
                                              [0,self.c],0,False)
                    # sequentially set the feature firing rates
                    fstep = (self.f_rate[1]-self.f_rate[0])/self.feature_layer
                    for i in range(self.feature_layer):
                        net['fire_rate'][fLayer][i] = self.f_rate[0]+fstep*(i+1)
                        
                    # create the excitatory and inhibitory connections
                    
                    # Excitatory weights
                    idx = blitnet.addWeights(net,iLayer,fLayer,[0,1],self.p_exc,self.n_init, False)
                    ex_weight = []
                    ex_weight.append(idx)
                    
                    # Inhibitory weights
                    inh_weight = []
                    idx = blitnet.addWeights(net,iLayer,fLayer,[-1,0],self.p_inh,self.n_init,False)
                    inh_weight.append(idx)            
                    
                    # Set the spikes times for the input images
                    spikeTimes = []
                    for n, ndx in enumerate(self.spike_rates):
                        nz_indicies = np.nonzero(ndx)
                        tempspikes = ndx[nz_indicies[0]]
                        tempspikes[tempspikes>=1] = 0.999
                        spiketime = (n+1)+tempspikes
                        spike_neuron = (np.column_stack((spiketime,nz_indicies[0]))).tolist()
                        spikeTimes.extend(spike_neuron)
                    
                    spikeTimes = torch.from_numpy(np.array(spikeTimes))
                    # set input spikes
                    blitnet.setSpikeTimes(net,0,spikeTimes)
                    
                    # Train the input to feature layer
                    # Train the feature layer
                    for t in range(int(self.T/10)):
                        blitnet.runSim(net,10)
                        # anneal learning rates
                        if np.mod(t,10)==0:
                            pt = pow(float(self.T-t)/self.T,self.annl_pow)
                            net['eta_ip'][fLayer] = self.n_itp*pt
                            net['eta_stdp'][ex_weight[-1]] = self.n_init*pt
                            net['eta_stdp'][inh_weight[-1]] = -1*self.n_init*pt
                                
                    # Turn off learning between input and feature layer
                    net['eta_ip'][fLayer] = 0.0
                    if self.p_exc > 0.0: net['eta_stdp'][ex_weight[-1]] = 0.0
                    if self.p_inh > 0.0: net['eta_stdp'][inh_weight[-1]] = 0.0
                    
                    # Create and train the output layer with the feature layer
                    oLayer = blitnet.addLayer(net,[self.output_layer,1],0.0,0.0,0.0,0.0,0.0,False)

                    # Add the excitatory connections
                    idx = blitnet.addWeights(net,fLayer,oLayer,[0.0,1.0],1.0,self.n_init,False)
                    ex_weight.append(idx)
                    
                    # Add the inhibitory connections
                    idx = blitnet.addWeights(net,fLayer,oLayer,[-1.0,0.0],1.0,-self.n_init,False)
                    inh_weight.append(idx)
                    
                    # Output spikes for spike forcing (final layer)
                    out_spks = np.zeros([(self.output_layer),2])
                    append_spks = np.zeros([(self.output_layer),2])
            
                    for n in range(self.output_layer):
                        out_spks[n] = [(n)+1.9,n]
                        append_spks[n] = [(n)+1.9,n]
                        
                    if self.location_repeat != 0:
                        base_spks = np.copy(out_spks)
                        for n in range(1,self.location_repeat):
                            base_spks[:,0] = base_spks[:,0] + self.test_t
                            out_spks = np.append(out_spks,base_spks,axis=0)
                    
                    for n in range(self.epoch):
                        out_spks[:,0] += self.output_layer
            
                        append_spks= torch.from_numpy(np.concatenate((append_spks,out_spks),axis=0))
                    
                    # Set the output spikes (spike forcing)
                    append_spks[:,0] += self.T
                    blitnet.setSpikeTimes(net,oLayer,append_spks)
                    
                    # Train the feature to output layer            
                    for t in range(self.T):
                        blitnet.runSim(net,1)
                        # Anneal learning rates
                        if np.mod(t,10)==0:
                            pt = pow(float(self.T-t)/(self.T),self.annl_pow)
                            net['eta_ip'][oLayer] = self.n_itp*pt
                            net['eta_stdp'][ex_weight[-1]] = self.n_init*pt
                            net['eta_stdp'][inh_weight[-1]] = -1*self.n_init*pt

                    # Turn off learning
                    net['eta_ip'][oLayer] = 0.0
                    net['eta_stdp'][ex_weight[-1]] = 0.0
                    net['eta_stdp'][inh_weight[-1]] = 0.0

                    # Clear the network output spikes
                    blitnet.setSpikeTimes(net,oLayer,[])
                    
                    # Reset network details
                    net['set_spks'][0] = []
                    #net['rec_spks'] = [True,True,True]
                    net['sspk_idx'] = [0,0,0]
                    net['step_num'] = 0
                    net['spikes'] = [[],[],[]]
                    
                    # check if output dir exsist, create if not
                    if not path.isdir(self.training_out):
                        os.mkdir(self.training_out)
                        
                    # Output the trained network
                    outputPkl = self.training_out + str(ensemble_num) + '.pkl'
                    with open(outputPkl, 'wb') as f:
                        pickle.dump(net, f)
                    
                    breakflag = False
                    # when ensemble training is done, pickle entire ensemble into a dictionary
                    if self.ensemble_max == (ensemble_num+1)*self.number_training_images or ensemble_num == range(self.number_ensembles)[-1]:
                        for n in range((self.ensemble_netNum-1)*breakUp,(self.ensemble_netNum*breakUp)):
                            pickleName = self.training_out + str(n) + '.pkl'
                            # Select training images from list
                            net = []
                            if os.path.isfile(pickleName):
                                with open(pickleName, 'rb') as f:
                                     net = pickle.load(f)
                                if n == range((self.ensemble_netNum-1)*breakUp,(self.ensemble_netNum*breakUp))[0]:
                                    ensemble_net = net
                                    for m in range(oLayer+2):
                                        ensemble_net['W'][m] = torch.unsqueeze(ensemble_net['W'][m],-1)
                                        if m <= (len(ensemble_net['thr'])-1):
                                            ensemble_net['thr'][m] = torch.unsqueeze(ensemble_net['thr'][m],-1)
                                            ensemble_net['const_inp'][m] = torch.unsqueeze(ensemble_net['const_inp'][m],-1)
                                            ensemble_net['fire_rate'][m] = torch.unsqueeze(ensemble_net['fire_rate'][m],-1)
                                else:
                                    for m in range(oLayer+2):
                                        ensemble_net['W'][m] = torch.concat((ensemble_net['W'][m],torch.unsqueeze(net['W'][m],-1)),-1)
                                        if m <= (len(ensemble_net['thr'])-1):
                                            ensemble_net['thr'][m] = torch.concat((ensemble_net['thr'][m],torch.unsqueeze(net['thr'][m],-1)),-1)
                                            ensemble_net['const_inp'][m] = torch.concat((ensemble_net['const_inp'][m],torch.unsqueeze(net['const_inp'][m],-1)),-1)
                                            ensemble_net['fire_rate'][m] = torch.concat((ensemble_net['fire_rate'][m],torch.unsqueeze(net['fire_rate'][m],-1)),-1)
                                
                                # delete the individual pickled net
                                os.remove(pickleName)
                        # pickle the ensemble network
                        outputPkl = self.training_out + 'ensemble_net'+str(self.ensemble_netNum)+'.pkl'
                        with open(outputPkl, 'wb') as f:
                            pickle.dump(ensemble_net, f)
                            
                        self.ensemble_max += self.copy_max
                        self.ensemble_netNum += 1
                    
                    yield
                    
            print('Training the ensembles')
            with alive_bar(self.number_ensembles) as sbar:
                for i in train_start():
                    sbar()
        
        '''
     Run the testing network
     '''

    def networktester(self):
        '''
        Network tester functions
        '''
        # set the input spikes
        def set_spikes():
            spikeTimes = []

            for n, ndx in enumerate(self.spike_rates):
               nz_indicies = np.nonzero(ndx)
               tempspikes = ndx[nz_indicies[0]]
               tempspikes[tempspikes>=1] = 0.999
               spiketime = (n+1)+tempspikes
               spike_neuron = (np.column_stack((spiketime,nz_indicies[0]))).tolist()
               spikeTimes.extend(spike_neuron)
            
            spikeTimes = torch.from_numpy(np.array(spikeTimes))
            
            for m in netDict:
                print('Setting spikes')
                # set input spikes
                blitnet.setSpikeTimes(netDict[str(m)],0,spikeTimes)
                
                netDict[str(m)]['set_spks'][0] = torch.unsqueeze(netDict[str(m)]['set_spks'][0],-1)
                tempspikes = torch.clone(netDict[str(m)]['set_spks'][0])
                for n in range(int(self.ensemble_max/self.number_training_images)-1):
                    netDict[str(m)]['set_spks'][0] = torch.concat((netDict[str(m)]['set_spks'][0],tempspikes),-1)
        
        # calculate and plot distance matrices
        def plotit(netx,name):
            reshape_mat = np.reshape(netx,(self.test_t,int(self.train_img/self.location_repeat)))
            # plot the matrix
            fig = plt.figure()
            plt.matshow(reshape_mat,fig, cmap=plt.cm.gist_yarg)
            plt.colorbar(label="Spike amplitude")
            fig.suptitle("Similarity "+name,fontsize = 12)
            plt.xlabel("Query",fontsize = 12)
            plt.ylabel("Database",fontsize = 12)
            plt.show()
        
        # calculate PR curves
        
        
        # network validation using alternative place matching algorithms and P@R calculation
        def network_validator():
            # reload training images for the comparisons
            self.test_true= False # reset the test true flag
            self.test_imgs = self.ims.copy()
            self.dataPath = '/home/adam/data/hpc/'
            self.fullTrainPaths = []
            for n in self.locations:
                self.fullTrainPaths.append(self.trainingPath+n)
            self.loadImages()
            # run sum of absolute differences caluclation
            validate.SAD(self)
            
        '''
        Setup & running network tester
        '''
        
        # Alter network running parameters for testing
        self.epoch = 1 # Only run the network once
        self.location_repeat = 1 # One location repeat for testing
        self.test_true = True # Flag for multiple data functions
        
        # unpickle the network
        print('Unpickling the ensemble network')
        self.ensemble_netNum = len([entry for entry in os.listdir(self.training_out) if os.path.isfile(os.path.join(self.training_out, entry))])
        netDict = {}
        for n in range(self.ensemble_netNum):
            with open(self.training_out+'ensemble_net'+str(n+1)+'.pkl', 'rb') as f:
                 netDict[str(n)] = pickle.load(f)
             
        # Load the network training images and set the input spikes     
        print('Loading dataset images')
        self.loadNames = self.filteredNames
        self.loadImages()

        self.setSpikeRates(self.imgs['testing'],self.ids['testing'])
        set_spikes()

        #_net['rec_spks'] = [True,True,True]'=
        numcorrect = 0
        
        try:
            self.ensemble_max = self.copy_max
        except AttributeError:
            self.ensemble_max = self.ensemble_max
            
        for n in netDict:
            netDict[str(n)]['n_ensemble'] = int(self.ensemble_max/self.number_training_images)
            
        # Combine all the networks and set each input image sequentially  
        start = timeit.default_timer()
        avPeakMatch = np.array([])
        avPeakFail = np.array([])
        avAccurate = np.zeros(self.number_ensembles)
        ensembleNum = self.number_training_images-1
        ensembleInd = 0
        for t in range(self.test_t*self.number_ensembles):
            tonump = np.array([])
            for g in netDict:
                _net = netDict[str(g)]
                ensemble.runSim(_net,1)
                # output the index of highest amplitude spike
                tonump = np.append(tonump,np.reshape(_net['x'][-1].detach().cpu().numpy(),(self.test_t*int(self.ensemble_max/self.number_training_images),1),order='F'))
            
            nidx = np.argmax(tonump)
            if nidx == t:
                numcorrect += 1
                print('\033[32m'+"!Match! for image - "+self.ids['testing'][t]+': '+str(t)+' - '+str(nidx))
                avPeakMatch = np.append(avPeakMatch,np.max(tonump))
                avAccurate[ensembleInd] += 1
            else:
                print('\033[31m'+":( fail for image - "+self.ids['testing'][t]+': '+str(t)+' - '+str(nidx))
                avPeakFail = np.append(avPeakFail,np.max(tonump))
            
            if t == ensembleNum:
                ensembleNum += self.number_training_images
                ensembleInd += 1
            #if nidx == t:
             #   numcorrect += 1
        print('\033[0m'+"It took this long ",timeit.default_timer() - start)
        print("Number of correct places "+str((numcorrect/(self.test_t*self.number_ensembles))*100)+"%")
        avPeakMatch[avPeakMatch == 0] = np.nan
        avPeakFail[avPeakFail == 0] = np.nan
        print('Average spike amplitude of matches: '+str(np.nanmean(avPeakMatch)))
        print('Average spike amplitude of fails: '+str(np.nanmean(avPeakFail)))   
        # plot the similarity matrices for each location repetition
        append_mat = []
        for n in self.mat_dict:
            if int(n) != 0:
                append_mat = append_mat + self.mat_dict[str(n)]
            else:
                append_mat = np.copy(self.mat_dict[str(n)])
        plot_name = "training images"
        #plotit(append_mat,plot_name)
        #plotit(self.net_x,plot_name)
        
        # pickle the ground truth matrix
        reshape_mat = np.reshape(append_mat,(self.test_t,int(self.train_img/self.location_repeat)))
        boolval = reshape_mat > 0 
        GTsoft = boolval.astype(int)
        GT = np.zeros((self.test_t,self.test_t), dtype=int)

        for n in range(len(GT)):
            GT[n,n] = 1
        plot_name = "Similarity absolute ground truth"
        #fig = plt.figure()
        #plt.matshow(GT,fig, cmap=plt.cm.gist_yarg)
        #plt.colorbar(label="Spike amplitude")
        #fig.suptitle(plot_name,fontsize = 12)
        #plt.xlabel("Query",fontsize = 12)
        #plt.ylabel("Database",fontsize = 12)
        #plt.show()
        #with open('./output/GT.pkl', 'wb') as f:
         #   pickle.dump(GT, f)
        
        self.VPRTempo_correct = 100*self.numcorrect/self.test_t
        #print(self.VPRTempo_correct,'% correct')
        
        # Clear the network output spikes
        blitnet.setSpikeTimes(net,2,[])
        
        # Reset network details
        net['set_spks'][0] = []
        net['rec_spks'] = [True,True,True]
        net['sspk_idx'] = [0,0,0]
        net['step_num'] = 0
        net['spikes'] = [[],[],[]]
        
        # Load the testing images
        #self.test_true = True # Set image path to the testing images
        self.test_true = True
        self.loadImages()
        
        # Set input layer spikes as the testing images
        set_spikes()
        
        # run the test netowrk    
        test_network()
        
        # store and print out number of correctly identified places
        self.VPRTempo_correct = 100*self.numcorrect/self.test_t
        
        # plot the similarity matrices for each location repetition
        append_mat = []
        for n in self.mat_dict:
            if int(n) != 0:
                append_mat = append_mat + self.mat_dict[str(n)]
            else:
                append_mat = np.copy(self.mat_dict[str(n)])
        plot_name = "VPRTempo"
        #plotit(append_mat,plot_name)
        #plotit(self.net_x,plot_name)
        # pickle the ground truth matrix
        S_in = np.reshape(append_mat,(self.test_t,int(self.train_img/self.location_repeat)))
        with open('./output/S_in.pkl', 'wb') as f:
            pickle.dump(S_in, f)
                
        # calculate the precision of the system
        self.precision = self.tp/(self.tp+self.fp)
        self.recall = self.tp/self.test_t
        #P, R = createPR(S_in, GT, GT)
        # plot PR curve
        #fig = plt.figure()
        #plt.plot(R,P)
        #fig.suptitle("VPRTempo Precision Recall curve",fontsize = 12)
        #plt.xlabel("Recall",fontsize = 12)
        #plt.ylabel("Precision",fontsize = 12)
        #plt.show()
        
        # plot spikes if they were recorded
        #if net['rec_spks'][0] == True:
         #   blitnet.plotSpikes(net,0)
        
        # clear the CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # if validation is set to True, run comparison methods
        if self.validation:
           network_validator()
        
        sadcorrect = self.sad_correct
        return self.numcorrect, append_mat, sadcorrect

'''
Run the network
'''        
if __name__ == "__main__":
    model = snn_model() # Instantiate model
    model.train()
    model.networktester()
    # unpickle the network
    print('Unpickling the ensemble network')
    with open(outfold+'ensemble_net.pkl', 'rb') as f:
         ensemble_net = pickle.load(f)
    
    # unpickle the image names
    print('Unpickling the image training names')
    if not bool(trained_imgs):
        with open(outfold+'img_ids.pkl', 'rb') as f:
             trained_imgs = pickle.load(f)
    
    # run supervised testing
    def supervisedTest():
        global totCorrect, network_mat, sadCorrect
        totCorrect = 0
        sadCorrect = 0
        network_mat = np.zeros(3300*3300)
        for t in ensemble_range:
            filter_names = trained_imgs[str(t)]
            correct = []
            correct, mat, sad = model.networktester(ensemble_net[str(t)],filter_names)
            totCorrect = totCorrect + correct
            sadCorrect = sadCorrect + sad
            #tempmat = np.zeros(625*132)
            #start = (len(mat)*t)
            #end = (len(mat)*(t+1))
            #tempmat[start:end] = mat
            #network_mat[82500*t:82500*(t+1)] = tempmat
            yield
    
    print('Testing ensemble network')
    with alive_bar(n_ensembles) as sbar:
        for i in supervisedTest():
            sbar()
    
    #reshape_mat = np.reshape(network_mat,(3300,3300))
   # plot_name = "Similarity VPRTempo"
    #fig = plt.figure()
    #plt.matshow(reshape_mat,fig, cmap=plt.cm.gist_yarg)
    #plt.colorbar(label="Pixel intensity")
    #fig.suptitle(plot_name,fontsize = 12)
    #plt.xlabel("Query",fontsize = 12)
    #plt.ylabel("Database",fontsize = 12)
    #plt.show()
    print('P@100R '+str(100*(totCorrect/3300)))
    print('P@100R for SAD '+str(100*(sadCorrect/3300)))