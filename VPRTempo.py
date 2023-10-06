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

class SNNLayer(nn.Module):
    def __init__(self, previous_layer=None,dims=[0,0,0],thr_range=[0,0], 
                 fire_rate=[0,0],ip_rate=0,stdp_rate=0,const_inp=[0,0],p=[1,1],
                 assign_weight=False,spk_force=False):
        super(SNNLayer, self).__init__()
        # Configure the network
        configure(self) # Sets the testing configuration
        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Check constraints etc
        if np.isscalar(thr_range): thr_range = [thr_range, thr_range]
        if np.isscalar(fire_rate): fire_rate = [fire_rate, fire_rate]
        if np.isscalar(const_inp): const_inp = [const_inp, const_inp]
        
        # Initialize Tensors
        self.dim = torch.tensor(dims, dtype=torch.int)
        self.x = torch.zeros(dims, device=self.device)
        self.x_prev = torch.zeros(dims, device=self.device)
        self.x_calc = torch.zeros(dims, device=self.device)
        self.x_input = torch.zeros(dims, device=self.device)
        self.x_fastinp = torch.zeros(dims, device=self.device)
        self.eta_ip = torch.tensor(ip_rate, device=self.device)
        self.eta_stdp = torch.tensor(stdp_rate, device=self.device)
        
        # Initialize Parameters
        self.thr = nn.Parameter(torch.zeros(dims, device=self.device).uniform_(thr_range[0], thr_range[1]))
        self.fire_rate = torch.zeros(dims, device=self.device).uniform_(fire_rate[0], fire_rate[1])
        
        # Sequentially set the feature firing rates (if any)
        if not torch.all(self.fire_rate==0).item():
            fstep = (fire_rate[1]-fire_rate[0])/dims[-1]
            
            # loop through all modules and feature layer neurons
            for x in range(self.number_modules):
                for i in range(dims[-1]):
                   self.fire_rate[x][:,i] = fire_rate[0]+fstep*(i+1)
                   
        self.have_rate = torch.any(self.fire_rate[:,:,0] > 0.0).to(self.device)
        self.const_inp = torch.zeros(dims, device=self.device).uniform_(const_inp[0], const_inp[1])
        self.p = p
        self.dims = dims
        
        # Additional State Variables
        self.set_spks = []
        self.sspk_idx = 0
        self.spikes = torch.empty([], dtype=torch.float64)
        self.spk_force = spk_force
        
        # Weights (if applicable)
        if assign_weight:
            excW, inhW, self.havconnExc, self.havconnInh = bn.addWeights(p=self.p,
                                                 dims=[previous_layer.dims[2],
                                                       dims[2]],
                                                 num_modules=self.number_modules)
            
            self.excW = nn.Parameter(excW)
            self.inhW = nn.Parameter(inhW)
        
class SNNTrainer(nn.Module):
    def __init__(self):
        super(SNNTrainer, self).__init__()
        # Configure the network
        configure(self) # Sets the testing configuration
        image_csv(self) # Defines images to load
        
        # Set the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set up the input layer
        self.input_layer = SNNLayer(dims=[self.number_modules,1,self.input])
        
        # Set up the feature layer
        self.feature_layer = SNNLayer(previous_layer=self.input_layer,
                                      dims=[self.number_modules,1,self.feature],
                                      thr_range=[0,0.5],
                                      fire_rate=[0.2,0.9],
                                      ip_rate=0.15,
                                      stdp_rate=0.005,
                                      const_inp=[0,0.1],
                                      p=[0.1,0.5],
                                      assign_weight=True)
        
        # Set up the output layer
        self.output_layer = SNNLayer(previous_layer=self.feature_layer,
                                    dims=[self.number_modules,1,self.output],
                                    ip_rate=0.15,
                                    stdp_rate=0.005,
                                    assign_weight=True,
                                    spk_force=True)
        
        # Define number of layers (will run training on all layers)
        self.layers = {'layer0':self.input_layer,
                       'layer1':self.feature_layer,
                       'layer2':self.output_layer}

    
    def train_model(self, train_loader, logger):
        
        # If using CUDA, run a dummy torch.bmm to 'spool-up' operations
        if self.device.type == "cuda":
            ut.dummy_bmm(self.device)

        warmup_iters = 10
        for _ in range(warmup_iters):
            _ = next(iter(train_loader))

        # Run the training for each layer, using defined parameters
        for layer in range(len(self.layers)-1):

            # Define in and out layer (2 layers trained at a time)
            in_layer = self.layers['layer'+str(layer)]
            out_layer = self.layers['layer'+str(layer+1)]
            
            # Output the learning rates for annealment during training
            n_initstdp = out_layer.eta_stdp.detach() # STDP
            n_initip = out_layer.eta_ip.detach() # ITP

            # Initialize the tqdm progress bar
            pbar = tqdm(total=int(self.T * self.epoch),
                        desc="Training layer "+str(layer)+" with layer "+str(layer+1),
                        position=0)
            
            # Run the training for the input to feature layer for specified epochs
            for epoch in range(self.epoch):
                mod = 0 # Used to determine the learning rate annealment
                for images, labels in train_loader: 

                    # Put input images on device (CPU, CUDA)
                    images = images.to(self.device)
                    
                    # Set spikes from input images
                    make_spikes = SetImageAsSpikes(self.intensity, test=False)
                    spikes = make_spikes(images)
                    
                    # Put labels on device (CPU, CUDA)
                    labels = labels.to(self.device)
                    idx = labels/self.filter
                    
                    # Layers don't include loaded input, calculate network spikes for up to training layers
                    if layer != 0:
                        spikes = bn.testSim(self.layers,layer,spikes)
                    
                    # Run one timestep of the training input to feature layer
                    bn.runSim(in_layer, out_layer, spikes, idx)
                    
                    # Anneal the learning rate
                    if np.mod(mod,10)==0:
                        pt = pow(float(self.T-mod)/self.T,self.annl_pow)
                        out_layer.eta_ip = torch.mul(n_initip,pt)
                        out_layer.eta_stdp = torch.mul(n_initstdp,pt)
                    mod += 1
                    
                    # Reset x and x_input for next iteration
                    out_layer.x.fill_(0.0)
                    out_layer.x_input.fill_(0.0)
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
        
        # Set up the input layer
        self.input_layer = SNNLayer(dims=[self.number_modules,1,self.input])
        
        # Set up the feature layer
        self.feature_layer = SNNLayer(previous_layer=self.input_layer,
                                      dims=[self.number_modules,1,self.feature],
                                      assign_weight=True)
        
        # Set up the output layer
        self.output_layer = SNNLayer(previous_layer=self.feature_layer,
                                    dims=[self.number_modules,1,self.output],
                                    assign_weight=True)
        
        # Define number of layers (will run training on all layers)
        self.layers = {'layer0':self.input_layer,
                       'layer1':self.feature_layer,
                       'layer2':self.output_layer}
        
        self.is_quantized = False
        
    def load_model(self,model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()

    def quantize_tensors(self):
        if not self.is_quantized:
            tensors_to_quantize = ["excW", "inhW", "thr"]

            # Iterate over the layers
            for layer_name, layer in self.layers.items():
                for tensor_name in tensors_to_quantize:
                    # Check if the tensor exists in the current layer
                    if hasattr(layer, tensor_name):
                        tensor = getattr(layer, tensor_name)

                        # Convert nn.Parameter to regular tensor if necessary
                        if isinstance(tensor, nn.Parameter):
                            tensor = tensor.data
                            # Explicitly delete the attribute
                            delattr(layer, tensor_name)

                        # If the tensor is all zeros, skip quantization
                        if (tensor == 0).all():
                            continue

                        # Check if all values in the tensor are negative, and if so, multiply by -1
                        if (tensor < 0).all():
                            tensor = tensor * -1

                        # Calculate the non-zero max and min of this tensor
                        non_zero_max = tensor[tensor.nonzero(as_tuple=True)].max()
                        non_zero_min = tensor[tensor.nonzero(as_tuple=True)].min()

                        # Calculate scale based on non-zero max and min
                        scale = (non_zero_max - non_zero_min) / 255.0

                        # Now, quantize the tensor
                        quantized_tensor = nn.quantized.Quantize(scale=scale, zero_point=0, dtype=torch.quint8)(tensor)
                        setattr(layer, tensor_name, quantized_tensor)



    def forward(self, test_loader):
        
        # If using CUDA, run a dummy torch.bmm to 'spool-up' operations
        if self.device.type == "cuda":
            ut.dummy_bmm(self.device)
            
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
            #images = images.to(self.device)
            #labels = labels.to(self.device)
            
            # Set the spikes for the input, tiling across modules if applicable
            make_spikes = SetImageAsSpikes(intensity=self.intensity,
                                           modules=self.number_modules)
            spikes = make_spikes(images)
            
            # Run the test sim
            out = bn.testSim(self.layers,len(self.layers)-1,spikes)
            
            # Store output and determine if output matches GT
            tonump = np.array([])
            tonump = np.append(tonump,np.reshape(out.cpu().numpy(),
                                        [1,1,int(self.number_training_images)]))
            if np.argmax(tonump) == idx:
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
    qconfig = torch.quantization.QConfig(
    activation=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MinMaxObserver),
    weight=torch.quantization.default_weight_fake_quant)

    model.qconfig = qconfig
    model = quantization.prepare_qat(model)

    # Generate model name, check if pre-trained model exists
    model_name = ("VPRTempo"+ # main name
                  str(model.input)+ # number input neurons
                  str(model.feature)+ # number feature neurons
                  str(model.output)+ # number output neurons
                  str(model.number_modules)+ # number of modules
                  '.pth')
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
                                          modules=model.number_modules,
                                          test=False)
        
        # Define the training dataloader class
        train_loader = DataLoader(train_dataset, 
                                  batch_size=model.number_modules, 
                                  shuffle=False,
                                  num_workers=8,
                                  persistent_workers=True)
        
        # Initialize, run, and save the training model
        trainer = SNNTrainer()
        trainer.train_model(train_loader,model.logger)
        trainer.save_model(os.path.join('./models',model_name))
    
    with torch.no_grad():  # Disable gradient computation during testing
        
        model = quantization.convert(model.eval())
        
        # Load the trained model into SNNModel()
        model.load_model(os.path.join('./models',model_name))    
        model = model.cpu()
        model.quantize_tensors()

        # Define the custom testing image dataset class
        test_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                          img_dirs=model.testing_dirs,
                                          transform=image_transform,
                                          skip=model.filter,
                                          max_samples=model.number_testing_images,
                                          modules=model.number_modules)
        
        # Define the testing dataloader class
        test_loader = DataLoader(test_dataset, 
                                  batch_size=1, 
                                  shuffle=False,
                                  num_workers=8,
                                  persistent_workers=True)
        model.forward(test_loader)


        for name, module in model.named_modules():
            if 'quantized' in str(type(module)):
                print(f"{name}: {type(module)} is quantized!")
