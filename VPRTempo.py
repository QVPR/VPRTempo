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

import os
import torch
import gc
import sys
sys.path.append('./src')
sys.path.append('./models')
sys.path.append('./settings')
sys.path.append('./output')
sys.path.append('./dataset')
sys.path.append('./config')
torch.multiprocessing.set_sharing_strategy("file_system")
import blitnet as bn
import utils as ut
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization

from config import configure, image_csv, model_logger
from dataset import CustomImageDataset, SetImageAsSpikes, ProcessImage
from torch.utils.data import DataLoader
from tqdm import tqdm

class SNNTrainer(nn.Module):
    def __init__(self):
        super(SNNTrainer, self).__init__()
        # Configure the network
        configure(self) # Sets the testing configuration
        image_csv(self) # Defines images to load
        
        # Set the device
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # Set up the feature layer
        self.feature_layer = bn.SNNLayer(
                                      dims=[self.input,self.feature],
                                      thr_range=[0,0.5],
                                      fire_rate=[0.2,0.9],
                                      ip_rate=0.15,
                                      stdp_rate=0.005,
                                      const_inp=[0,0.1],
                                      p=[0.1,0.5]
                                      )
        
        # Set up the output layer
        self.output_layer = bn.SNNLayer(
                                    dims=[self.feature,self.output],
                                    ip_rate=0.15,
                                    stdp_rate=0.005,
                                    spk_force=True
                                    )
        
        # Define number of layers (will run training on all layers)
        self.layers = {
                       'layer0':self.feature_layer,
                       'layer1':self.output_layer
                       }

    
    def train_model(self, train_loader, logger):
        
        # If using CUDA, run a dummy torch.bmm to 'spool-up' operations
        if self.device.type == "cuda":
            ut.dummy_bmm(self.device)
        
        # Warm up the DataLoader
        warmup_iters = 10
        for _ in range(warmup_iters):
            _ = next(iter(train_loader))
        
        # Run the training for each layer, using defined parameters
        for lyrCount, lyr in enumerate(self.layers):
            
            # Get the training layer
            layer = self.layers[lyr]
            
            # Output the learning rates for annealment during training
            n_initstdp = layer.eta_stdp.detach() # STDP
            n_initip = layer.eta_ip.detach() # ITP
            
            # Initialize the tqdm progress bar
            pbar = tqdm(total=int(self.T * self.epoch),
                        desc="Training "+str(lyr),
                        position=0)
            
            # Run the training for the input to feature layer for specified epochs
            for epoch in range(self.epoch):
                # Used to determine the learning rate annealment, resets at each epoch
                mod = 0 
                
                # Load and run images through BliTNET
                for images, labels in train_loader: 
                    
                    # Reset fire rate to None
                    fr = None
                    
                    # Put input images on device (CPU, CUDA)
                    images = images.to(self.device)
                    
                    # Set spikes from input images
                    make_spikes = SetImageAsSpikes(self.intensity, test=False)
                    spikes = make_spikes(images)
                    
                    # Put labels on device (CPU, CUDA)
                    labels = labels.to(self.device)
                    idx = labels/self.filter
                    
                    # Layers don't include loaded input, calculate network spikes for up to training layers
                    if list(lyr[-1]) != ['0']:
                        blitnet = bn.BLiTNET(layer_lst=self.layers,
                                             spikes=spikes,
                                             idx=idx,
                                             testCount=lyrCount)
                        spikes = blitnet.testSim()
                        fr = self.layers['layer'+str(lyrCount-1)].fire_rate
                        
                    # Run one timestep of the training input to feature layer
                    blitnet = bn.BLiTNET(
                                        layer=layer,
                                        spikes=spikes,
                                        idx=idx,
                                        fr=fr
                                        )
                    blitnet.runSim()
                    
                    # Anneal the learning rate
                    if np.mod(mod,10)==0:
                        pt = pow(float(self.T-mod)/self.T,self.annl_pow)
                        layer.eta_ip = torch.mul(n_initip,pt)
                        layer.eta_stdp = torch.mul(n_initstdp,pt)
                    
                    # Update the learning rate annealment modifier
                    mod += 1
                    
                    # Reset x and x_input for next iteration
                    layer.x.fill_(0.0)
                    layer.x_input.fill_(0.0)
                    pbar.update(1)
  
            # Close the tqdm progress bar
            pbar.close()
            print('')
            
        # Clear the cache and garbage collect (if using cuda)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
                
    def save_model(self, model_out):    
        # save model
        torch.save(self.state_dict(), model_out)

class SNNModel(nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()

        # Configure the network
        configure(self) # Sets the testing configuration
        image_csv(self) # Defines images to load
        model_logger(self) # Sets the logger
        
        
        # Set up the feature layer
        self.feature_layer = bn.SNNLayer(dims=[self.input,self.feature])
        
        # Set up the output layer
        self.output_layer = bn.SNNLayer(dims=[self.feature,self.output])
        
        # Define number of layers (will run training on all layers)
        self.layers = {
                        'layer0':self.feature_layer,
                        'layer1':self.output_layer
                       }
        
    def load_model(self,model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()

    def forward(self, test_loader):
            
        idx=0
        numcorr = 0

          # If using CUDA, run a dummy torch.bmm to 'spool-up' operations
        if self.device.type == "cuda":
            ut.dummy_bmm(self.device)

        warmup_iters = 10
        for _ in range(warmup_iters):
            _ = next(iter(test_loader))
        
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.number_testing_images,
                    desc="Running the test network",
                    position=0)

        # Run test network for each individual input
        for images, labels in test_loader:
            
            # Set images to the specified device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Set the spikes for the input, tiling across modules if applicable
            make_spikes = SetImageAsSpikes(intensity=self.intensity)
            spikes = make_spikes(images)
            
            # Run the test sim
            blitnet = bn.BLiTNET(layer_lst=self.layers,
                                 spikes=spikes,
                                 testCount=len(self.layers)
                                 )
            out = blitnet.testSim()

            if torch.argmax(out.reshape(1, self.number_training_images)) == idx:
                numcorr += 1

            idx+=1
            
            # Update the progress bar
            pbar.update(1)
        
        pbar.close()
        model.logger.info('')
        model.logger.info("P@100R: "+
                     str(round((numcorr/self.number_testing_images)*100,2))+'%')
        
        # Clear cache and empty garbage (if applicable)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
if __name__ == "__main__":
    # Initialize the model and image transforms
    model = SNNModel()
    qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # Generate model name, check if pre-trained model exists
    model_name = ("VPRTempo"+ # main name
                  str(model.input)+ # number input neurons
                  str(model.feature)+ # number feature neurons
                  str(model.output)+ # number output neurons
                  str(model.number_modules)+ # number of modules
                  '.pth')
    
    # Check if a pre-trained model exists
    if os.path.exists(os.path.join('./models',model_name)):
        pretrain_flg = True
        
        # Prompt user to retrain network if desired
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        retrain = input(prompt)
        print('')
        
        # Retrain network, set flag to False
        if retrain == 'y':
            pretrain_flg = False
    else:
        # No pretrained model exists
        pretrain_flg = False
    
    # Define the image transform class
    image_transform = ProcessImage(model.dims,model.patches)
    
    # If no pre-existing model, train new model with set configuration
    if not pretrain_flg:
        # Define the custom training image dataset class
        train_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                          img_dirs=model.training_dirs,
                                          transform=image_transform,
                                          skip=model.filter,
                                          max_samples=model.number_training_images,
                                          test=False)
        
        # Define the training dataloader class
        train_loader = DataLoader(train_dataset, 
                                  batch_size=1, 
                                  shuffle=False,
                                  num_workers=1,
                                  persistent_workers=True)
        
        # Initialize, run, and save the training model
        trainer = SNNTrainer()
        trainer.to('cpu')

        trainer.feature_layer.qconfig = qconfig
        trainer.output_layer.qconfig = qconfig
        trainer = torch.quantization.prepare_qat(trainer, inplace=True)
        
        trainer.train_model(train_loader,model.logger)
        
        trainer.eval()
        trainer = torch.quantization.convert(trainer, inplace=True)
        
        trainer.save_model(os.path.join('./models',model_name))
    
    with torch.no_grad():  # Disable gradient computation during testing
        model.to('cpu')
        model.feature_layer.qconfig = qconfig
        model.output_layer.qconfig = qconfig
        model = torch.quantization.prepare(model, inplace=False)
        model = torch.quantization.convert(model, inplace=False)
        # Load the trained model into SNNModel()
        model.load_model(os.path.join('./models',model_name))    
        
        
        # Define the custom testing image dataset class
        test_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                          img_dirs=model.testing_dirs,
                                          transform=image_transform,
                                          skip=model.filter,
                                          max_samples=model.number_testing_images)
        
        # Define the testing dataloader class
        test_loader = DataLoader(test_dataset, 
                                  batch_size=1, 
                                  shuffle=False,
                                  num_workers=1,
                                  persistent_workers=True)
        model.forward(test_loader)