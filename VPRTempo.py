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
import random
import sys
sys.path.append('./src')
sys.path.append('./weights')
sys.path.append('./settings')
sys.path.append('./output')

import numpy as np
import BlitnetDense as blitnet
import blitnet_ensemble as ensemble
import validation as validate
import matplotlib.pyplot as plt

from os import path
from alive_progress import alive_bar
from metrics import createPR


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
        self.number_training_images =25 # alter number of training images
        self.number_testing_images = 1000 # alter number of testing images
        self.number_modules = 1 # number of module networks
        self.location_repeat = 2 # Number of training locations that are the same
        self.locations = ["fall","spring"] # which datasets are used in the training
        self.test_location = "summer"
        
        # Image and patch normalization settings
        self.imWidth = 28 # image width for patch norm
        self.imHeight = 28 # image height for patch norm
        self.num_patches = 7 # number of patches
        self.intensity = 255 # divide pixel values to get spikes in range [0,1]
        
        # Network and training settings
        self.input_layer = (self.imWidth*self.imHeight) # number of input layer neurons
        self.feature_layer = int(self.input_layer*1) # number of feature layer neurons
        self.output_layer = int(self.number_training_images/self.number_modules) # number of output layer neurons (match training images)
        self.train_img = self.output_layer # number of training images
        self.epoch = 4 # number of training iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda": # clear cuda cache, initialize, and syncgronize gpu
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()
            torch.cuda.init()
            torch.cuda.synchronize(device=self.device)
        self.T = int((self.number_training_images/self.number_modules)*self.location_repeat) # number of training steps
        self.annl_pow = 2 # learning rate anneal power
        self.filter = 8 # filter images every 8 seconds (equivelant to 8 images)
        
        # Select training images from list
        with open('./nordland_imageNames.txt') as file:
            self.imageNames = [line.rstrip() for line in file]
            
        # Filter the loading images based on self.filter
        self.filteredNames = []
        for n in range(0,len(self.imageNames),self.filter):
            self.filteredNames.append(self.imageNames[n])
        del self.filteredNames[self.number_training_images:len(self.filteredNames)]
        
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
                 
    # sets the spike rates from imported images - convert pixel range [0,255] to [0,1]
    # spike rates are set as a 3D tensor with dimensions (m x r x s)
    # m = module
    # r = location repeat
    # s = spikes for number of training images
    def setSpikeRates(self, images, ids):   
         
         # output loaded images from numpy array to tensor
         data = torch.tensor(images,device=self.device)
         n_input = self.imWidth * self.imHeight 
         
         # Set the spike rates based on the number of example training images
         
         # loop through data and append spike rates for each image
         # organise into 3D tensor based on number of expert modules
         if self.test_true: # if loading testing data (repeat input across modules)
             self.init_rates = torch.empty(0,device=self.device) 
             for m in range(self.number_testing_images):
  
                 self.init_rates = torch.cat((self.init_rates, torch.reshape(data[m,:,:]/self.intensity,(n_input,)))) 
                 
             self.init_rates = torch.unsqueeze(self.init_rates,-1)    
             # define the initial spike rates
             for o in range(self.number_modules):
                 if o == 0:
                     self.spike_rates = torch.unsqueeze(self.init_rates,0)
                 else:
                     self.spike_rates = torch.concat((self.spike_rates,torch.unsqueeze(self.init_rates,0)),0)

                            
         else: # if loading training data, have separate inputs across modules
            for o in range(self.number_modules):
                start = []
                end = []
                for j in range(self.location_repeat):
                    mod = j * self.number_training_images
                    start.append(int(self.number_training_images/self.number_modules)*(o) + mod)
                    end.append(int(self.number_training_images/self.number_modules)*(o + 1) + mod)
    
                # define the initial spike rates for location repeats
                for m in range(self.location_repeat):
                    init_rates = torch.empty(0,device=self.device)
                    for jdx, j in enumerate(range(start[m],end[m])):    
                        init_rates = torch.cat((init_rates,
                                   torch.reshape(data[j,:,:]/self.intensity,(n_input,))),0)  
                    if m == 0:
                        self.init_rates = torch.unsqueeze(init_rates,-1)
                    else:
                        self.init_rates = torch.cat((self.init_rates,torch.unsqueeze(init_rates,-1)),0)
                
                # output spike rates into modules
                if o == 0: # append the location repeat initial spikes to a new module
                    self.spike_rates = torch.unsqueeze(self.init_rates,0)
                else:
                    self.spike_rates = torch.concat((self.spike_rates,torch.unsqueeze(self.init_rates,0)),0)

             
    def checkTrainTest(self):
        # Check if pre-trained network exists, prompt if retrain or run
        if path.isdir(self.training_out):
            retrain = input("A network with these parameters exists, re-train network? (y/n):\n")
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
              
            if os.path.isfile(self.training_out + 'net.pkl'):
                os.remove(self.training_out+'net.pkl')
                os.remove(self.training_out+'GT_imgnames.pkl')
            
            def train_start():  
                    
                print('Loading training images')
                
                self.loadNames = self.filteredNames
                # store the ground truth image names
                # check if output dir exsist, create if not
                if not path.isdir(self.training_out):
                    os.mkdir(self.training_out)
                outputPkl = self.training_out + 'GT_imgnames.pkl'
                with open(outputPkl, 'wb') as f:
                    pickle.dump(self.loadNames, f)
                self.loadImages()
                self.setSpikeRates(self.imgs['training'],self.ids['training'])
                
                # create new network
                print('Creating and setting network up')
                net = blitnet.newNet(self.number_modules,self.imWidth*self.imHeight)
                iLayer = blitnet.addLayer(net,[self.number_modules,self.input_layer,1],0.0,0.0,0.0,0.0,0.0,
                                                                     False)   
                fLayer = blitnet.addLayer(net,[self.number_modules,self.feature_layer,1],[0,self.theta_max],
                                          [self.f_rate[0],self.f_rate[1]],self.n_itp,
                                          [0,self.c],0,False)
                
                # sequentially set the feature firing rates
                fstep = (self.f_rate[1]-self.f_rate[0])/self.feature_layer
                for x in range(self.number_modules):
                    for i in range(self.feature_layer):
                        net['fire_rate'][fLayer][x][i] = self.f_rate[0]+fstep*(i+1)
                    
                # create the excitatory and inhibitory connections
                idx = blitnet.addWeights(net,iLayer,fLayer,[-1,0,1],[self.p_exc,self.p_inh],self.n_init, False)
                weight = []
                weight.append(idx-1)
                weight.append(idx)        
                
                '''
                Feature layer training
                '''
                # Set the spikes times for the input images
                net['set_spks'][0] = self.spike_rates
                layers = [len(net['W'])-2, len(net['W'])-1, len(net['W_lyr'])-1]
                # Train the input to feature layer
                # Train the feature layer
                for epoch in range(self.epoch):
                    net['step_num'] = 0
                    for t in range(int(self.T)):
                        blitnet.runSim(net,1,self.device,layers)
                        # anneal learning rates
                        if np.mod(t,10)==0:
                            pt = pow(float(self.T-t)/self.T,self.annl_pow)
                            net['eta_ip'][fLayer] = self.n_itp*pt
                            net['eta_stdp'][weight[0]] = self.n_init*pt
                            net['eta_stdp'][weight[1]] = -1*self.n_init*pt
                
                # refresh CUDA cache (if available) for output layer training
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                # Turn off learning between input and feature layer
                net['eta_ip'][fLayer] = 0.0
                if self.p_exc > 0.0: net['eta_stdp'][weight[0]] = 0.0
                if self.p_inh > 0.0: net['eta_stdp'][weight[1]] = 0.0
                
                # get the feature spikes for training the output layer
                net['x_feat'] = []
                net['step_num'] = 0
                for t in range(int(self.T)):
                    blitnet.runSim(net,1,self.device,layers)
                    net['x_feat'].append(net['x'][1])
                # Create and train the output layer with the feature layer
                oLayer = blitnet.addLayer(net,[self.number_modules,self.output_layer,1],0.0,0.0,0.0,0.0,0.0,False)

                # Add excitatory and inhibitory connections
                idx = blitnet.addWeights(net,fLayer,oLayer,[-1.0,0.0,1.0],[1.0,1.0],self.n_init,False)
                weight.append(idx)

                # Output spikes for spike forcing (final layer)
                out_spks = torch.tensor([0],device=self.device,dtype=float)

                '''
                Output layer training
                '''
                net['spike_dims'] = 1
                layers = [len(net['W'])-2, len(net['W'])-1, len(net['W_lyr'])-1]
                # Train the feature to output layer   
                for epoch in range(self.epoch): # number of training epochs
                    net['step_num'] = 0
                    for t in range(self.T):
                        out_spks = torch.tensor([0],device=self.device,dtype=float)
                        net['set_spks'][-1] = torch.tile(out_spks,(self.number_modules,self.output_layer,1))
                        blitnet.runSim(net,1,self.device,layers)
                        # Anneal learning rates
                        if np.mod(t,10)==0:
                            pt = pow(float(self.T-t)/(self.T),self.annl_pow)
                            net['eta_ip'][oLayer] = self.n_itp*pt
                            net['eta_stdp'][2] = self.n_init*pt
                            net['eta_stdp'][3] = -1*self.n_init*pt
                        if int(self.T/2) - 1 == t:
                            net['step_num'] = 0
                # Turn off learning
                net['eta_ip'][oLayer] = 0.0
                net['eta_stdp'][2] = 0.0
                net['eta_stdp'][3] = 0.0

                # Clear the network output spikes
                net['set_spks'][-1] = []
                net['spike_dims'] = self.input_layer
                # Reset network details
                net['sspk_idx'] = [0,0,0]
                net['step_num'] = 0
                net['spikes'] = [[],[],[]]
                    
                # Output the trained network
                outputPkl = self.training_out + 'net.pkl'
                with open(outputPkl, 'wb') as f:
                    pickle.dump(net, f)
             
                
                    #yield
            
            #print('Training the expert modules')
            #with alive_bar(self.number_modules) as sbar:
             #  for i in train_start():
              #     sbar()

            train_start()

        '''
     Run the testing network
     '''

    def networktester(self):
        '''
        Network tester functions
        '''
        
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
        print('Unpickling the network')
        with open(self.training_out+'net.pkl', 'rb') as f:
             net = pickle.load(f)
             
        # Load the network training images and set the input spikes     
        print('Loading dataset images')
        random.shuffle(self.filteredNames)
        self.loadNames = self.filteredNames[0:self.number_testing_images]
        self.loadImages()
        
        # set the spike rates
        self.setSpikeRates(self.imgs['testing'],self.ids['testing'])
        net['set_spks'][0] = self.spike_rates
        
        # unpickle the ground truth image names
        with open(self.training_out+'GT_imgnames.pkl', 'rb') as f:
             GT_imgnames = pickle.load(f)
        
        # set number of correct places to 0
        numcorrect = 0
        net['spike_dims'] = self.input_layer
        for t in range(self.number_testing_images):
            tonump = np.array([])
            blitnet.testSim(net,device=self.device)
            # output the index of highest amplitude spike
            tonump = np.append(tonump,np.reshape(net['x'][-1].detach().cpu().numpy(),(self.number_training_images,1),order='F'))
            gt_ind = GT_imgnames.index(self.loadNames[t])
            nidx = np.argmax(tonump)

            if nidx == gt_ind:
               numcorrect += 1
        print('\033[0m'+"It took this long ",timeit.default_timer() - start)
        print("Number of correct places "+str((numcorrect/(self.test_t*self.number_ensembles))*100)+"%")
        avPerc = 100*(avAccurate/self.number_training_images)
        ensemblename = []
        for n in range(self.number_ensembles):
            ensemblename.append(str(n+1))
            
        fig = plt.figure()
        plt.bar(ensemblename,avPerc)
        plt.xlabel("Ensemble number")
        plt.ylabel("P@100R (%)")
        plt.title("Performance of individual ensemble networks")
        plt.xticks(rotation=45)
        plt.tick_params(axis='x', which='major',labelsize=7)
        plt.tight_layout()
        plt.show()
        
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