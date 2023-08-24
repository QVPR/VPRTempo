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
import gc
import shutil
import wandb
import pprint

import sys
sys.path.append('./src')
sys.path.append('./weights')
sys.path.append('./settings')
sys.path.append('./output')

import blitnet as bn
import utils as ut
import numpy as np


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
        self.number_training_images =3300 # alter number of training images
        self.number_testing_images = 3300# alter number of testing images
        self.number_modules = 20 # number of module networks
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
        self.feature_layer = int(self.input_layer*2) # number of feature layer neurons
        self.output_layer = int(self.number_training_images/self.number_modules) # number of output layer neurons
        self.epoch = 4 # number of training iterations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda": # clear cuda cache, initialize, and syncgronize gpu
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            torch.cuda.set_device(self.device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.init()
            torch.cuda.synchronize(device=self.device)
        self.T = int((self.number_training_images/self.number_modules)
                             *self.location_repeat) # number of training steps
        self.annl_pow = 2 # learning rate anneal power
        self.imgs = {'training':[],'testing':[]}
        self.ids = {'training':[],'testing':[]}
        self.spike_rates = {'training':[],'testing':[]}
        self.hpsweep = False
        
        # Hyperparamters
        self.n_init = 0.005
        self.n_itp = 0.15
        self.theta_max = 0.5
        self.c = 0.1
        
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
 
        # Network weights name
        self.training_out = './weights/'+str(self.input_layer)+'i'+\
                                            str(self.feature_layer)+\
                                        'f'+str(self.output_layer)+\
                                            'o'+str(self.epoch)+'/'
    
    def initialize(self,condition):
        
        if condition == 'testing':
            self.test_true = True
            #random.shuffle(self.filteredNames)
            del self.filteredNames[self.number_testing_images:len(self.filteredNames)]
            self.epoch = 1 # Only run the network once
            self.location_repeat = 1 # One location repeat for testing
            imgNum = self.number_testing_images
        else:
            imgNum = self.number_training_images
            
        # load the training images
        self.imgs[condition], self.ids[condition] = ut.loadImages(self.test_true,
                                            self.fullTrainPaths,
                                            self.filteredNames,
                                            [self.imWidth,self.imHeight],
                                            self.num_patches,
                                            self.testPath,
                                            self.test_location)
        
        self.spike_rates[condition] = ut.setSpikeRates(self.imgs[condition],
                                            self.ids[condition],
                                            self.device,
                                            [self.imWidth,self.imHeight],
                                            self.test_true,
                                            imgNum,
                                            self.number_modules,
                                            self.intensity,
                                            self.location_repeat)
    
    '''
    Run the training network
    '''
    def train(self,f_rateL,f_rateH,p_exc,p_inh):
        self.epoch = 4
        # remove contents of the weights folder
        if os.path.isfile(self.training_out + 'net.pkl'):
            os.remove(self.training_out+'net.pkl')
        if os.path.isfile(self.training_out + 'GT_imgnames.pkl'):
            os.remove(self.training_out+'GT_imgnames.pkl')
        if os.path.isdir(self.training_out+'images/training/'):
            shutil.rmtree(self.training_out+'images/training/')
        if not os.path.isdir(self.training_out):
            os.mkdir(self.training_out)
        '''
        Network startup and initialization
        '''

        # create a new blitnet netowrk
        net = bn.newNet(self.number_modules,self.imWidth*self.imHeight)
        
        # add the input layer
        bn.addLayer(net,[self.number_modules,1,self.input_layer],
                             0.0,0.0,0.0,0.0,0.0,False)   
        
        # add the feature layer
        bn.addLayer(net,[self.number_modules,1,self.feature_layer],
                        [0,self.theta_max],
                        [f_rateL,f_rateH],
                         self.n_itp,
                        [0,self.c],
                         0,
                         False)
        
        # sequentially set the feature firing rates for the feature layer
        fstep = (f_rateH-f_rateL)/self.feature_layer
        
        # loop through all modules and feature layer neurons
        for x in range(self.number_modules):
            for i in range(self.feature_layer):
                net['fire_rate'][1][x][:,i] = f_rateL+fstep*(i+1)
            
        # add excitatory inhibitory connections for input and feature layer
        bn.addWeights(net,0,1,[-1,0,1],[p_exc,p_inh],self.n_init)
        
        # begin timer for network training
        # Set the spikes times for the input images
        net['set_spks'][0] = self.spike_rates['training']
        layers = [len(net['W'])-2, len(net['W'])-1, len(net['W_lyr'])-1]
        # Train the input to feature layer
        # Train the feature layer
        for epoch in range(self.epoch):
            net['step_num'] = 0
            for t in range(int(self.T)):
                bn.runSim(net,1,self.device,layers)
                torch.cuda.empty_cache()
                # anneal learning rates
                if np.mod(t,10)==0:
                    pt = pow(float(self.T-t)/self.T,self.annl_pow)
                    net['eta_ip'][1] = self.n_itp*pt
                    net['eta_stdp'][0] = self.n_init*pt
                    net['eta_stdp'][1] = -1*self.n_init*pt
        
        # Turn off learning between input and feature layer
        net['eta_ip'][1] = 0.0
        if p_exc > 0.0: net['eta_stdp'][0] = 0.0
        if p_inh > 0.0: net['eta_stdp'][1] = 0.0
        
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
        
        # Output spikes for spike forcing (final layer)
        out_spks = torch.tensor([0],device=self.device,dtype=float)
        
        net['spike_dims'] = 1 # change spike dims for output spike indexing
        net['set_spks'][0] = [] # remove input spikes
        layers = [len(net['W'])-2, len(net['W'])-1, len(net['W_lyr'])-1]
        
        # Train the feature to output layer   
        for epoch in range(self.epoch):
            net['step_num'] = 0
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
        
        # Output the trained network
        outputPkl = self.training_out + 'net.pkl'
        with open(outputPkl, 'wb') as f:
            pickle.dump(net, f)
            
        # output the ground truth image names for later testing
        outputPkl = self.training_out + 'GT_imgnames.pkl'
        with open(outputPkl, 'wb') as f:
            pickle.dump(self.filteredNames, f)
   
    '''
     Run the testing network
     '''

    def networktester(self):
        self.epoch=1 
        '''
        Network startup and initialization
        '''

        # unpickle the network
        with open(self.training_out+'net.pkl', 'rb') as f:
             net = pickle.load(f)

        net['set_spks'][0] = self.spike_rates['testing']
        
        # unpickle the ground truth image names
        with open(self.training_out+'GT_imgnames.pkl', 'rb') as f:
             GT_imgnames = pickle.load(f)
        
        # set number of correct places to 0
        numcorrect = 0
        r_n5 = 0
        net['spike_dims'] = self.input_layer
        numpconc = np.array([])
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
        
        p100r = round((numcorrect/self.number_training_images)*100,2)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return p100r
'''
Run the network
'''        
if __name__ == "__main__":
    # load model
    model = snn_model()
    
    # log into weights & biases
    wandb.login()
    
    # define the method and parameters for grid search
    sweep_config = {'method':'grid'}
    metric = {'name':'p100r', 'goal':'maximize'}
    
    sweep_config['metric'] = metric
    
    parameters_dict = {   
                       'f_rateL' :
                           {'values' : [0.05,0.1,0.2]},
                       'f_rateH' :
                           {'values' : [0.8,0.9,1.0]},
                       'p_exc' :
                           {'values' : [0.1,0.2,0.3,0.4,0.5]},
                       'p_inh' :
                           {'values' : [0.1,0.2,0.3,0.4,0.5]}
                           }
    
    sweep_config['parameters'] = parameters_dict
    pprint.pprint(sweep_config)
    
    # start sweep controller
    sweep_id = wandb.sweep(sweep_config, project="vprtempo-grid-search")
    
    # load the testing and training data for VPRTempo
    print('Getting images and spikes for training data')
    model.initialize('training')
    print('Getting images and spikes for testing data')
    model.initialize('testing')
    
    # test that model works prior to running wandb
   # print('Testing the network training prior to sweep')
   # model.train(0.25,0.012,0.2,0.25,0.125,0.025,0.125,0.01)
   # print('Running test data')
   # p100r = model.networktester()
    
    print('Starting the w&b sweeping')
    # initialize w&b
    def wandsearch(config=None):
        with wandb.init(config=config):
            config = wandb.config
            model.train(config.f_rateL,config.f_rateH,config.p_exc,config.p_inh)
            
            p100r = model.networktester()
            wandb.log({"p100r" : p100r})
            
    wandb.agent(sweep_id,wandsearch)