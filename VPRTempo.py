#MIT License

#Copyright (c) 2023 Adam Hines

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
import torch
import random
import sys
sys.path.append('./src')
sys.path.append('./weights')
sys.path.append('./settings')
sys.path.append('./output')

import numpy as np
import blitnet_open as blitnet


from os import path
from alive_progress import alive_bar


'''
Spiking network model class
'''
class snn_model():
    def __init__(self):
        super().__init__()
        
        # Image and patch normalization settings
        self.dataPath = '/home/adam/data/VPRTempo_training/training_data/' # training datapath
        self.testPath = '/home/adam/data/VPRTempo_training/summer_gamma/' # 
        self.imWidth = 28 # image width for patch norm
        self.imHeight = 28 # image height for patch norm
        self.num_patches = 7 # number of patches
        self.intensity = 255 # divide pixel values to get spikes in range [0,1]
        
        # Network and training settings
        self.input_layer = 784 # number of input layer neurons
        self.feature_layer = int(self.input_layer*7)# number of feature layer neurons
        self.output_layer = 400 # number of output layer neurons (match training images)
        self.train_img = 400 # number of training images
        self.epoch = 4 # number of training iterations
        self.test_t = 200 # number of testing time points
        self.cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # saliency calculating on cpu or gpu
        self.T = self.train_img*self.epoch # number of training steps
        self.run_time = self.T*2+self.test_t
        self.annl_pow = 2 # learning rate anneal power
        
        # Hyperparamters
        self.theta_max = 0.25 # maximum threshold value
        self.n_init = 0.05 # initial learning rate value
        self.n_itp = 0.25 # initial intrinsic threshold plasticity rate[[0.9999]]
        self.f_rate = [0.0012,0.2]# firing rate range
        self.p_exc = 0.025 # probability of excitatory connection
        self.p_inh = 0.125 # probability of inhibitory connection
        self.c= 0.125 # constant input 
        self.pickler = True # default True, pickles network details and can be reused without retraining
        
        # Test settings 
        self.test_true = False # leave default to False
        
        # Print network details
        print('////////////')
        print('Running VPRTempo v0.1 - Queensland University of Technology, Centre for Robotics')
        print('\\\\\\\\\\\\\\\\\\\\\\\\')
        print('Theta: '+str(self.theta_max))
        print('Initial learning rate: '+str(self.n_init))
        print('ITP Learning: '+str(self.n_itp))
        print('Firing rate: '+str(self.f_rate[0]) +' to '+ str(self.f_rate[1]))
        print('Excitatory p: '+str(self.p_exc))
        print('Inhibitory p: '+str(self.p_inh))
        print('Constant input '+str(self.c))
        
        # select training images from list
        with open('./nordland_imageNames.txt') as file:
            lines = [line.rstrip() for line in file]
        self.img_load = lines
  
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
            
        # resize image to 28x28 and patch normalize        
        self.img = cv2.resize(self.img,(self.imWidth, self.imHeight))
        self.patch_normalise_pad() 
        self.img = np.uint8(255.0 * (1 + self.im_norm) / 2.0)

        
    # Image loader function - runs all image import functions
    def loadImages(self):
            
         if self.test_true:
            self.dataPath = self.testPath
        
         self.ims = []
         self.ids = []
         print('Loading images and patch normalising')

         for m in self.img_load:
             self.fullpath = self.dataPath+m
             # read and convert image from BGR to RGB 
             self.img = cv2.imread(self.fullpath)[:,:,::-1]
             # convert image
             self.img = cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
             self.processImage()
             self.ims.append(self.img)
             self.ids.append(m)
        
        # pickle image ids to keep track of shuffling    
         if self.test_true:
            with open('./output/test_ids.pkl','wb') as f:
                pickle.dump(self.ids,f)
         else:
            with open('./output/train_ids.pkl','wb') as f:
                pickle.dump(self.ids,f)
                
         data = {'x': np.array(self.ims), 'y': np.array(self.ids), 
                             'rows': self.imWidth, 'cols': self.imHeight}
         num_testing_imgs = data['x'].shape[0]
         self.num_examples = num_testing_imgs
         n_input = self.imWidth * self.imHeight 
         
         self.spike_rates = []
         self.init_rates = []
         print('Converting images to spikes')
         for jdx, j in enumerate(range(int(self.num_examples))):    
             self.init_rates.append((data['x'][j%num_testing_imgs,:,:].reshape((n_input))/self.intensity))  

         for n in range(int(self.epoch)): 

             self.spike_rates.extend(self.init_rates)

         if not self.test_true:
             self.spike_rates.extend(self.spike_rates)
    
    '''
    Run the training network
    '''
    def run_train(self):
        
        # network weights name
        training_out = './weights/'+str(self.input_layer)+'i'+\
                                            str(self.feature_layer)+\
                                        'f'+str(self.output_layer)+\
                                            'o'+str(self.epoch)+'.pkl'
                                        
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
                yield
            
            spikeTimes = np.array(spikeTimes)
            # set input spikes
            blitnet.setSpikeTimes(net,0,spikeTimes)
            
        def train_feature():     
            # Train the feature layer
            for t in range(int(self.T/10)):
                blitnet.runSim(net,10)
                # anneal learning rates
                if np.mod(t,10)==0:
                    pt = pow(float(self.T-t)/self.T,self.annl_pow)
                    net['eta_ip'][fLayer] = self.n_itp*pt
                    net['eta_stdp'][ex_weight[-1]] = self.n_init*pt
                    net['eta_stdp'][inh_weight[-1]] = -1*self.n_init*pt
                yield
        
        def train_output():
            # Train the layer
            for t in range(self.T):
                
                blitnet.runSim(net,1)
                # Anneal learning rates
                if np.mod(t,10)==0:
                    pt = pow(float(self.T-t)/(self.T),self.annl_pow)
                    net['eta_ip'][oLayer] = self.n_itp*pt
                    net['eta_stdp'][ex_weight[-1]] = self.n_init*pt
                    net['eta_stdp'][inh_weight[-1]] = -1*self.n_init*pt
                yield
        
        if not path.exists(training_out):
            
            self.loadImages()
            
            # create new network
            print("Creating network layers")
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
            idx = blitnet.addWeights(net,iLayer,fLayer,[0,1],self.p_exc,self.n_init,
                                                                             False)
            ex_weight = []
            ex_weight.append(idx)
            inh_weight = []

            idx = blitnet.addWeights(net,iLayer,fLayer,[-1,0],self.p_inh,self.n_init,
                                                                             False)
            inh_weight.append(idx)            
            
            print('Setting spike times')
            with alive_bar(len(self.spike_rates)) as sbar:
                for i in set_spikes():
                    sbar()
                        
            # Turn off learning
            net['eta_ip'][fLayer] = 0.0
            if self.p_exc > 0.0: net['eta_stdp'][ex_weight[-1]] = 0.0
            if self.p_inh > 0.0: net['eta_stdp'][inh_weight[-1]] = 0.0
            
            oLayer = blitnet.addLayer(net,[self.output_layer,1],0.0,0.0,0.0,0.0,0.0,
                                                                 False)
        
            # Train the output layer
            # Add the excitatory and balancing inhib connections
            idx = blitnet.addWeights(net,fLayer,oLayer,[0.0,1.0],1.0,self.n_init,False)
            ex_weight.append(idx)
            idx = blitnet.addWeights(net,fLayer,oLayer,[-1.0,0.0],1.0,-self.n_init,False)
            inh_weight.append(idx)
            
            # Output spikes for spike forcing (final layer)
            out_spks = np.zeros([(self.output_layer),2])
            append_spks = np.zeros([(self.output_layer),2])

            for n in range(self.output_layer):
                out_spks[n] = [(n)+1.5,n]
                append_spks[n] = [(n)+1.5,n]
                
            for n in range(self.epoch):
                out_spks[:,0] += self.output_layer

                append_spks= np.concatenate((append_spks,out_spks),axis=0)
            
            # Set the output spikes (spike forcing)
            append_spks[:,0] += self.T
            
            blitnet.setSpikeTimes(net,oLayer,append_spks)
                        
            print('Training the output layer')
            with alive_bar(self.T) as outbar:
                for i in train_output():
                    outbar()
            blitnet.plotSpikes(net,0)
            # Turn off learning
            net['eta_ip'][oLayer] = 0.0
            net['eta_stdp'][ex_weight[-1]] = 0.0
            net['eta_stdp'][inh_weight[-1]] = 0.0
            
            # Clear the network output spikes
            blitnet.setSpikeTimes(net,oLayer,[])
            if self.pickler:
                print('Pickling trained network')
                with open(training_out, 'wb') as f:
                    pickle.dump(net, f)
            print('Training done in '+str(self.T)+' computations')
            
        else:
            print('Unpickling '+training_out)
            with open(training_out, 'rb') as f:
                net = pickle.load(f)
                
        '''
         Run the testing network
         '''
        self.epoch = 1
        def networktester(self):

            self.loadImages()
            
            print('Setting spike times')
            with alive_bar(len(self.spike_rates)) as sbar:
                for i in set_spikes():
                    sbar()
            
            net['rec_spks'] = [True,True,True]
            net['sspk_idx'] = [0,0,0]
            net['step_num'] = 0
            net['spikes'] = [[],[],[]]
            
            # load the training and testing IDs for correct matching
            with open('./output/train_ids.pkl', 'rb') as f:
                train_ids = pickle.load(f)      
            if self.train_true:
                with open('./output/traintest_ids.pkl', 'rb') as f:
                    test_ids = pickle.load(f)  
            if self.test_true:
                with open('./output/test_ids.pkl', 'rb') as f:
                    test_ids = pickle.load(f)  
                
            def test_network():
                # Test the output
                self.correct_idx = []
                self.numcorrect = 0
                for t in range(self.test_t):
                    blitnet.runSim(net,1)
                    nidx = np.argmax(net['x'][-1]) 
                    if nidx < 200:
                        if  train_ids[nidx] == test_ids[t] or train_ids[(nidx+200)] == test_ids[t]:
                            self.numcorrect = self.numcorrect+1
                            self.correct_idx.append(t)
                    else:
                        if train_ids[nidx] == test_ids[t] or train_ids[(nidx-200)] == test_ids[t]:
                            self.numcorrect = self.numcorrect+1
                            self.correct_idx.append(t)
                   
                    yield
    
            with alive_bar(self.test_t) as testbar:
                for i in test_network():
                    testbar()
                    
            spkforc = 100*self.numcorrect/self.test_t
            print(spkforc,'% correct')
            
            # If using grid searching function, pick the conditions that give 
            # the best performance
            if self.grid_searching and self.test_true:
                if self.settings['optimisation'] == 'thetaconst':
                    output_path = './output/thetaconst.pkl'
                    with open(output_path, 'rb') as f:
                        thetaconst = pickle.load(f)
                    
                    if spkforc > thetaconst[2]:
                        condition = [self.theta_max,self.c,spkforc]
                        with open(output_path, 'wb') as f:
                            pickle.dump(condition, f)
                
                elif self.settings['optimisation'] == 'firing':
                    output_path = './output/firing.pkl'
                    with open(output_path, 'rb') as f:
                        firing = pickle.load(f)
                    
                    if spkforc > firing[2]:
                        condition = [self.f_rate[0],self.f_rate[1],spkforc]
                        with open(output_path, 'wb') as f:
                            pickle.dump(condition, f)
                    
                elif self.settings['optimisation'] == 'excinh':
                    output_path = './output/excinh.pkl'
                    with open(output_path, 'rb') as f:
                        excinh = pickle.load(f)
                    
                    if spkforc > excinh[2]:
                        condition = [self.p_exc,self.p_inh,spkforc]
                        with open(output_path, 'wb') as f:
                            pickle.dump(condition, f)
                
                else:
                    output_path = './output/learningrate.pkl'
                    with open(output_path, 'rb') as f:
                        learningrate = pickle.load(f)
                    
                    if spkforc > learningrate[2]:
                        condition = [self.n_init,self.n_itp,spkforc]
                        with open(output_path, 'wb') as f:
                            pickle.dump(condition, f)
        
        #print('Running training dataset on trained network')
        with open('./nordland_testNames.txt') as file:
            lines = [line.rstrip() for line in file]
        self.img_load = lines
        self.train_true = True
        networktester(self)
        blitnet.plotSpikes(net,0)
        self.train_true = False

        self.test_true = True
        print('Running testing dataset on trained network')

        random.shuffle(self.img_load)
        networktester(self)
        blitnet.plotSpikes(net,0)
        pause=1

'''
Run the network
'''        
if __name__ == "__main__":
    model = snn_model()
    model.run_train()