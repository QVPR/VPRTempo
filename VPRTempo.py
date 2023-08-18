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

import pickle
import os
import torch
import timeit
import random
import shutil
import sys
sys.path.append('./src')
sys.path.append('./weights')
sys.path.append('./settings')
sys.path.append('./output')

import blitnet as bn
import utils as ut
import numpy as np
import validation as val
import matplotlib.pyplot as plt

from os import path
from metrics import createPR
from halo import Halo


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
        self.testPath = '/home/adam/data/testing_data/'  # testing datapath
        self.number_training_images =1000 # alter number of training images
        self.number_testing_images = 100# alter number of testing images
        self.number_modules = 50 # number of module networks
        self.location_repeat = 2 # Number of training locations that are the same
        self.locations = ["fall","spring"] # which datasets are used in the training
        self.test_location = "summer"
        self.filter = 8 # filter images every 8 seconds (equivelant to 8 images)
        
        '''
        NETWORK SETTINGS
        '''
        # Image and patch normalization settings
        self.imWidth = 28 # image width for patch norm
        self.imHeight = 28 # image height for patch norm
        self.num_patches = 7 # number of patches
        self.intensity = 255 # divide pixel values to get spikes in range [0,1]
        
        # Network and training settings
        self.input_layer = (self.imWidth*self.imHeight) # number of input layer neurons
        self.feature_layer = int(self.input_layer*1) # number of feature layer neurons
        self.output_layer = int(self.number_training_images/self.number_modules) # number of output layer neurons
        self.epoch = 5 # number of training iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda": # clear cuda cache, initialize, and syncgronize gpu
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()
            torch.cuda.init()
            torch.cuda.synchronize(device=self.device)
        self.T = int((self.number_training_images/self.number_modules)
                             *self.location_repeat) # number of training steps
        self.annl_pow = 2 # learning rate anneal power
        
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
        
        '''
        DATA SETTINGS
        '''
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
        
        # Print network details
        print('////////////')
        print('VPRTempo - Temporally Encoded Visual Place Recognition v1.1.0-alpha')
        print('Queensland University of Technology, Centre for Robotics')
        print('')
        print('© 2023 Adam D Hines, Peter G Stratton, Michael Milford, Tobias Fischer')
        print('MIT license - https://github.com/QVPR/VPRTempo')
        print('\\\\\\\\\\\\\\\\\\\\\\\\')
        print('')
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
    
    # Check if pre-trained network exists, prompt if retrain or run        
    def checkTrainTest(self):
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        print('')
        if path.isdir(self.training_out):
            retrain = input(prompt)
        else:
            retrain = 'y'
        return retrain
    
    '''
    Run the training network
    '''
    def train(self):

        # check if training required
        check = self.checkTrainTest()
        
        # rerun network
        if check == 'y':
            
            # remove contents of the weights folder
            if os.path.isfile(self.training_out + 'net.pkl'):
                os.remove(self.training_out+'net.pkl')
            if os.path.isfile(self.training_out + 'GT_imgnames.pkl'):
                os.remove(self.training_out+'GT_imgnames.pkl')
            if os.path.isdir(self.training_out+'images/training/'):
                shutil.rmtree(self.training_out+'images/training/')
                
            '''
            Network startup and initialization
            '''
            print('')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Network startup and initialization')    
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('')
            print('Loading training images')
            
            # load the training images
            self.imgs, self.ids = ut.loadImages(self.test_true,
                                                self.fullTrainPaths,
                                                self.filteredNames,
                                                [self.imWidth,self.imHeight],
                                                self.num_patches,
                                                self.testPath,
                                                self.test_location)
            
            print('Setting spike rates from loaded images')
            # calculate input spikes from training images
            self.spike_rates = ut.setSpikeRates(self.imgs['training'],
                                                self.ids['training'],
                                                self.device,
                                                [self.imWidth,self.imHeight],
                                                self.test_true,
                                                self.number_training_images,
                                                self.number_modules,
                                                self.intensity,
                                                self.location_repeat)
            
            print('Creating network and setting weights')
            # create a new blitnet netowrk
            net = bn.newNet(self.number_modules,self.imWidth*self.imHeight)
            
            # add the input layer
            bn.addLayer(net,[self.number_modules,1,self.input_layer],
                                 0.0,0.0,0.0,0.0,0.0,False)   
            
            # add the feature layer
            bn.addLayer(net,[self.number_modules,1,self.feature_layer],
                            [0,self.theta_max],
                            [self.f_rate[0],self.f_rate[1]],
                             self.n_itp,
                            [0,self.c],
                             0,
                             False)
            
            # sequentially set the feature firing rates for the feature layer
            fstep = (self.f_rate[1]-self.f_rate[0])/self.feature_layer
            
            # loop through all modules and feature layer neurons
            for x in range(self.number_modules):
                for i in range(self.feature_layer):
                    net['fire_rate'][1][x][:,i] = self.f_rate[0]+fstep*(i+1)
                
            # add excitatory inhibitory connections for input and feature layer
            bn.addWeights(net,0,1,[-1,0,1],[self.p_exc,self.p_inh],self.n_init)
            init_weights = [net['W'][0].clone().detach(),net['W'][1].clone().detach()]
            
            '''
            Feature layer training
            '''
            print('')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Training the input to feature layer')    
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('')
            # begin timer for network training
            start = timeit.default_timer()
            # Set the spikes times for the input images
            net['set_spks'][0] = self.spike_rates
            layers = [len(net['W'])-2, len(net['W'])-1, len(net['W_lyr'])-1]
            # Train the input to feature layer
            # Train the feature layer
            for epoch in range(self.epoch):
                net['step_num'] = 0
                epochStart = timeit.default_timer()
                for t in range(int(self.T)):
                    bn.runSim(net,1,self.device,layers)
                    torch.cuda.empty_cache()
                    # anneal learning rates
                    if np.mod(t,10)==0:
                        pt = pow(float(self.T-t)/self.T,self.annl_pow)
                        net['eta_ip'][1] = self.n_itp*pt
                        net['eta_stdp'][0] = self.n_init*pt
                        net['eta_stdp'][1] = -1*self.n_init*pt
                print('Epoch '+str(epoch+1)+' trained in: '
                      +str(round(timeit.default_timer()-epochStart,2))+'s')
                print('')
                
            print('Finished training input to feature layer')
            
            '''
            Preparations for feature to output layer training
            '''
            
            # refresh CUDA cache (if available) for output layer training
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Turn off learning between input and feature layer
            net['eta_ip'][1] = 0.0
            if self.p_exc > 0.0: net['eta_stdp'][0] = 0.0
            if self.p_inh > 0.0: net['eta_stdp'][1] = 0.0
            
            print('Getting feature layer spikes for output layer training')
            # get the feature spikes for training the output layer
            net['x_feat'] = []
            net['step_num'] = 0
            for t in range(int(self.T)): # run blitnet without learning to get feature spikes
                bn.runSim(net,1,self.device,layers)
                net['x_feat'].append(net['x'][1]) # dictionary output of feature spikes
                
            print('Creating output layer')    
            # Create and train the output layer with the feature layer
            bn.addLayer(net,[self.number_modules,1,self.output_layer],
                        0.0,0.0,0.0,0.0,0.0,False)

            # Add excitatory and inhibitory connections
            bn.addWeights(net,1,2,[-1.0,0.0,1.0],[1.0,1.0],self.n_init)
            init_weights.append(net['W'][2].clone().detach())
            init_weights.append(net['W'][3].clone().detach())
            
            # Output spikes for spike forcing (final layer)
            out_spks = torch.tensor([0],device=self.device,dtype=float)

            '''
            Output layer training
            '''
            print('')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Training the feature to output layer')    
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('')
            
            net['spike_dims'] = 1 # change spike dims for output spike indexing
            net['set_spks'][0] = [] # remove input spikes
            layers = [len(net['W'])-2, len(net['W'])-1, len(net['W_lyr'])-1]
            
            # Train the feature to output layer   
            for epoch in range(self.epoch):
                net['step_num'] = 0
                epochStart = timeit.default_timer()
                for t in range(self.T):
                    out_spks = torch.tensor([0],device=self.device,dtype=float)
                    net['set_spks'][-1] = torch.tile(out_spks,
                                                     (self.number_modules,
                                                      1,
                                                      self.output_layer))
                    net['x'][1] = net['x_feat'][t]
                    bn.runSim(net,1,self.device,layers)
                    # Anneal learning rates
                    if np.mod(t,10)==0:
                        pt = pow(float(self.T-t)/(self.T),self.annl_pow)
                        net['eta_ip'][2] = self.n_itp*pt
                        net['eta_stdp'][2] = self.n_init*pt
                        net['eta_stdp'][3] = -1*self.n_init*pt
                    if int(self.T/2) - 1 == t:
                        net['step_num'] = 0     
                print('Epoch '+str(epoch+1)+' trained in: '
                      +str(round(timeit.default_timer()-epochStart,2))+'s')
                print('')
                
            print('Finished training feature to output layer')
            print('')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Network trained in '+str(round(timeit.default_timer()-start,2))
                                                  +'s')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('')
            
            # Turn off learning
            net['eta_ip'][2] = 0.0
            net['eta_stdp'][2] = 0.0
            net['eta_stdp'][3] = 0.0

            # Clear the network output spikes
            net['set_spks'][-1] = []
            net['spike_dims'] = self.input_layer
            # Reset network details
            net['sspk_idx'] = [0,0,0]
            net['step_num'] = 0
            net['spikes'] = [[],[],[]]
            net['x'] = [[],[],[]]
            net['x_feat'] = []
            
            print('Validating network training')
            
            # set input spikes for training data from one location
            net['set_spks'][0] = ut.setSpikeRates(self.imgs['training'],
                                                self.ids['training'],
                                                self.device,
                                                [self.imWidth,self.imHeight],
                                                True,
                                                self.number_training_images,
                                                self.number_modules,
                                                self.intensity,
                                                1)
            
            # correct place matches variable
            numcorrect = 0
            outconcat = np.array([])
            
            # run the test network on training data to evaluate network performance
            start = timeit.default_timer()
            for t in range(self.number_training_images):
                tonump = np.array([])
                bn.testSim(net,device=self.device)
                # output the index of highest amplitude spike
                tonump = np.append(tonump,
                                   np.reshape(net['x'][-1].cpu().numpy(),
                                  [1,1,int(self.number_training_images)]))
                nidx = np.argmax(tonump)
                outconcat = np.append(outconcat,tonump)
                
                gt_ind = self.filteredNames.index(self.filteredNames[t])
                
                # adjust number of correct matches if GT matches peak output
                if gt_ind == nidx:
                   numcorrect += 1
                   
            end = timeit.default_timer()
            
            # network must match >95% of training places to be succesfull
            p100r = (numcorrect/self.number_training_images)*100
            testFlag = (p100r>95)
            if testFlag: # network training was successfull
                
                print('')
                print('Network training successful!')
                print('')
                
                print('Perfomance details:')
                print("-------------------------------------------")
                print('P@100R: '+str(p100r)+
                      '%  |  Query frequency: '+
                      str(round(self.number_training_images/(end-start),2))+'Hz')
                
                print('Trained on '+self.locations[0]+'/'+self.locations[1]+
                      '  |  Queried with '+self.locations[0])
                print('')
                
                # create output folder (if it does not already exist)
                if not os.path.isdir(self.training_out):
                    os.mkdir(self.training_out)
                if not os.path.isdir(self.training_out+'images/'):
                    os.mkdir(self.training_out+'images/')
                if not os.path.isdir(self.training_out+'images/training/'):
                    os.mkdir(self.training_out+'images/training/')
                
                # plot the similarity matrix for network validation
                concatReshape = np.reshape(outconcat,
                                           (self.number_training_images,
                                            self.number_training_images))
                
                plot_name = "Similarity: network training validation"
                ut.plot_similarity(concatReshape,plot_name,
                                   self.training_out+'images/training/')
                
                # Reset network details
                net['sspk_idx'] = [0,0,0]
                net['step_num'] = 0
                net['spikes'] = [[],[],[]]
                net['x'] = [[],[],[]]
                net['set_spks'][0] = 0
                
               # plot the weight matrices
                cmap = plt.cm.magma
                
                print('Plotting weight matrices')
                # initial weights
                ut.plot_weights(init_weights[0],'Initial I->F Excitatory Weights',
                                cmap,self.number_modules/5,0.5,self.training_out)
                ut.plot_weights(init_weights[1], 'Initial I->F Inhibitory Weights',
                                cmap,self.number_modules/5,0.1,self.training_out)
                ut.plot_weights(init_weights[2], 'Initial F->O Excitatory Weights',
                                cmap,self.number_modules,0.1,self.training_out)
                ut.plot_weights(init_weights[3], 'Initial F->O Inhibitory Weights',
                                cmap,self.number_modules,0.1,self.training_out)
                
                # calculated weights
                ut.plot_weights(net['W'][0],'I->F Excitatory Weights',
                                cmap,self.number_modules/5,0.5,self.training_out)
                ut.plot_weights(net['W'][1], 'I->F Inhibitory Weights',
                                cmap,self.number_modules/5,0.1,self.training_out)
                ut.plot_weights(net['W'][2],'F->O Excitatory Weights',
                                cmap,self.number_modules,0.1,self.training_out)
                ut.plot_weights(net['W'][3], 'F->O Inhibitory Weights',
                                cmap,self.number_modules,0.1,self.training_out)
                
                print('Network formatting and saving...') 
                
                # Output the trained network
                outputPkl = self.training_out + 'net.pkl'
                with open(outputPkl, 'wb') as f:
                    pickle.dump(net, f)
                    
                # output the ground truth image names for later testing
                outputPkl = self.training_out + 'GT_imgnames.pkl'
                with open(outputPkl, 'wb') as f:
                    pickle.dump(self.filteredNames, f)
                
                print('Network succesfully saved!')


        '''
     Run the testing network
     '''

    def networktester(self):
         
        '''
        Network startup and initialization
        '''
        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Testing startup and initialization')    
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('')
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
        self.filteredNames[0:self.number_testing_images]
        
        print('Setting spike times from loaded images')
        # load the testing images
        self.imgs, self.ids = ut.loadImages(self.test_true,
                                            self.fullTrainPaths,
                                            self.filteredNames,
                                            [self.imWidth,self.imHeight],
                                            self.num_patches,
                                            self.testPath,
                                            self.test_location)
        
        # set the spike rates
        # calculate input spikes from training images
        self.spike_rates = ut.setSpikeRates(self.imgs['testing'],
                                            self.ids['testing'],
                                            self.device,
                                            [self.imWidth,self.imHeight],
                                            self.test_true,
                                            self.number_testing_images,
                                            self.number_modules,
                                            self.intensity,
                                            self.location_repeat)
        net['set_spks'][0] = self.spike_rates
        
        # unpickle the ground truth image names
        with open(self.training_out+'GT_imgnames.pkl', 'rb') as f:
             GT_imgnames = pickle.load(f)
        
        print('')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Running test network')    
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('')
        
        # set number of correct places to 0
        numcorrect = 0
        r_n5 = 0
        net['spike_dims'] = self.input_layer
        numpconc = np.array([])
        start = timeit.default_timer()
        for t in range(self.number_testing_images):
            tonump = np.array([])
            bn.testSim(net,device=self.device)
            # output the index of highest amplitude spike
            tonump = np.append(tonump,np.reshape(net['x'][-1].cpu().numpy(),[1,1,int(self.number_training_images)]))
            numpconc = np.append(numpconc,tonump)
            gt_ind = GT_imgnames.index(self.filteredNames[t])
            nidx = np.argmax(tonump)
            nidxpart = np.argpartition(tonump,-5)[-5:]

            if gt_ind == nidx:
               numcorrect += 1
            if gt_ind in nidxpart:
                r_n5 += 1
        print('Number of correct matches P@100R - '+str((numcorrect/self.number_testing_images)*100)+'%')
        print('Number of correct matches R@N=5 - '+str((r_n5/self.number_testing_images)*100)+'%')
        end = timeit.default_timer()
        queryHertz = self.number_testing_images/(end-start)
        print('System queried at '+str(round(queryHertz,2))+'Hz')


'''
Run the network
'''        
if __name__ == "__main__":
    model = snn_model() # Instantiate model
    model.train() # Run network training (will check if already trained)
    model.networktester() # Test the network