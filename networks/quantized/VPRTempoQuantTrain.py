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
sys.path.append('./output')
sys.path.append('./dataset')

import blitnet as bn
import numpy as np
import torch.nn as nn
import torch.quantization as quantization
import torchvision.transforms as transforms

from dataset import CustomImageDataset, ProcessImage
from torch.utils.data import DataLoader
from torch.ao.quantization import QuantStub, DeQuantStub
from tqdm import tqdm

class VPRTempoQuantTrain(nn.Module):
    def __init__(self, args, dims, logger):
        super(VPRTempoQuantTrain, self).__init__()

        # Set the arguments
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
        setattr(self, 'dims', dims)
        
        # Only CPU available for quantization
        self.device = "cpu"

        # Set the logger
        self.logger = logger

        # Set the dataset file
        self.dataset_file = os.path.join('./dataset', self.dataset + '.csv')

        # Add quantization stubs for Quantization Aware Training (QAT)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()   

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        # Define layer architecture
        self.input = int(dims[0]*dims[1])
        self.feature = int(self.input * 2)
        self.output = int(args.num_places / args.num_modules)

        # Set the total timestep count
        self.location_repeat = len(args.database_dirs) # Number of times to repeat the locations
        self.T = int((self.num_places / self.num_modules) * self.location_repeat * self.epoch)

        """
        Define trainable layers here
        """
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature],
            thr_range=[0, 0.5],
            fire_rate=[0.2, 0.9],
            ip_rate=0.15,
            stdp_rate=0.005,
            p=[0.1, 0.5],
            device=self.device
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            ip_rate=0.15,
            stdp_rate=0.005,
            spk_force=True,
            p=[0.25, 0.75],
            device=self.device
        )
        
    def add_layer(self, name, **kwargs):
        """
        Dynamically add a layer with given name and keyword arguments.
        
        :param name: Name of the layer to be added
        :type name: str
        :param kwargs: Hyperparameters for the layer
        """
        # Check for layer name duplicates
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        
        # Add a new SNNLayer with provided kwargs
        setattr(self, name, bn.SNNLayer(**kwargs))
        
        # Add layer name and index to the layer_dict
        self.layer_dict[name] = self.layer_counter
        self.layer_counter += 1                           

    def _anneal_learning_rate(self, layer, mod, itp, stdp):
        """
        Anneal the learning rate for the current layer.
        """
        if np.mod(mod, 100) == 0: # Modify learning rate every 100 timesteps
            pt = pow(float(self.T - mod) / self.T, 2)
            layer.eta_ip = torch.mul(itp, pt) # Anneal intrinsic threshold plasticity learning rate
            layer.eta_stdp = torch.mul(stdp, pt) # Anneal STDP learning rate
            
        return layer

    def train_model(self, train_loader, layer, model, model_num, prev_layers=None):
        """
        Train a layer of the network model.

        :param train_loader: Training data loader
        :param layer: Layer to train
        :param prev_layers: Previous layers to pass data through
        """

        # Initialize the tqdm progress bar
        pbar = tqdm(total=int(self.T),
                    desc="Training ",
                    position=0)
        
        # Initialize the learning rates for each layer (used for annealment)
        init_itp = layer.eta_ip.detach()
        init_stdp = layer.eta_stdp.detach()
        mod = 0  # Used to determine the learning rate annealment, resets at each epoch
        # idx scale factor for different modules
        idx_scale = (self.max_module*self.filter)*model_num
        # Run training for the specified number of epochs
        for epoch in range(self.epoch):
            # Run training for the specified number of timesteps
            for spikes, labels in train_loader:
                spikes, labels = spikes.to(self.device), labels.to(self.device)
                idx = torch.round((labels - idx_scale) / self.filter) # Set output index for spike forcing
                # Pass through previous layers if they exist
                if prev_layers:
                    with torch.no_grad():
                        for prev_layer_name in prev_layers:
                            prev_layer = getattr(self, prev_layer_name) # Get the previous layer object
                            spikes = self.forward(spikes, prev_layer) # Pass spikes through the previous layer
                            spikes = bn.clamp_spikes(spikes, prev_layer) # Clamp spikes [0, 0.9]
                else:
                    prev_layer = None
                # Get the output spikes from the current layer
                pre_spike = spikes.detach() # Previous layer spikes for STDP
                spikes = self.forward(spikes, layer) # Current layer spikes
                spikes_noclp = spikes.detach() # Used for inhibitory homeostasis
                spikes = bn.clamp_spikes(spikes, layer) # Clamp spikes [0, 0.9]
                # Calculate STDP
                layer = bn.calc_stdp(pre_spike,spikes,spikes_noclp,layer, idx, prev_layer=prev_layer)
                # Adjust learning rates
                layer = self._anneal_learning_rate(layer, mod, init_itp, init_stdp)
                # Update the annealing mod & progress bar 
                mod += 1
                pbar.update(1)

        # Close the tqdm progress bar
        pbar.close()

    def forward(self, spikes, layer):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        
        spikes = self.quant(spikes)
        spikes = layer.w(spikes)
        spikes = self.dequant(spikes)
        
        return spikes
    
    def save_model(self, models, model_out):    
        """
        Save the trained model to models output folder.
        """
        state_dicts = {}
        for i, model in enumerate(models):  # Assuming models_list is your list of models
            state_dicts[f'model_{i}'] = model.state_dict()

        torch.save(state_dicts, model_out)
            
def generate_model_name_quant(model):
    """
    Generate the model name based on its parameters.
    """
    return ("VPRTempoQuant" +
            str(model.input) +
            str(model.feature) +
            str(model.output) +
            str(model.num_modules) +
            '.pth')

def check_pretrained_model(model_name):
    """
    Check if a pre-trained model exists and prompt the user to retrain if desired.
    """
    if os.path.exists(os.path.join('./models', model_name)):
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        retrain = input(prompt).strip().lower()
        return retrain == 'n'
    return False

def train_new_model_quant(models, model_name, qconfig):
    """
    Train a new model.

    :param model: Model to train
    :param model_name: Name of the model to save after training
    :param qconfig: Quantization configuration
    """
    # Initialize the image transforms and datasets
    image_transform = transforms.Compose([
        ProcessImage(models[0].dims, models[0].patches)
    ])
    # Automatically generate user_input_ranges
    user_input_ranges = []
    start_idx = 0

    for _ in range(models[0].num_modules):
        range_temp = [start_idx, start_idx+((models[0].max_module-1)*models[0].filter)]
        user_input_ranges.append(range_temp)
        start_idx = range_temp[1] + models[0].filter
    if models[0].num_places < models[0].max_module:
        max_samples=models[0].num_places
    else:
        max_samples = models[0].max_module
    # Keep track of trained layers to pass data through them
    # Training each layer
    trained_models = []
    for i, model in enumerate(models):
        trained_layers = [] 
        img_range=user_input_ranges[i]
        train_dataset = CustomImageDataset(annotations_file=models[0].dataset_file, 
                                base_dir=models[0].data_dir,
                                img_dirs=models[0].database_dirs,
                                transform=image_transform,
                                skip=models[0].filter,
                                test=False,
                                img_range=img_range,
                                max_samples=max_samples)
        # Initialize the data loader
        train_loader = DataLoader(train_dataset, 
                                batch_size=1, 
                                shuffle=True,
                                num_workers=8,
                                persistent_workers=True)
        model = quantization.prepare_qat(model, inplace=False)
        for layer_name, _ in sorted(models[0].layer_dict.items(), key=lambda item: item[1]):
            print(f"Training layer: {layer_name}")
            layer = (getattr(model, layer_name))

            # Train the layers
            model.train_model(train_loader, layer, model, i, prev_layers=trained_layers)
            trained_layers.append(layer_name) 
        trained_models.append(quantization.convert(model, inplace=False))
        # After training the current layer, add it to the list of trained layer

    # Save the model
    model.save_model(trained_models,os.path.join('./models', model_name))   