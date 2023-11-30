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
import subprocess
import sys
sys.path.append('./src')
sys.path.append('./models')
sys.path.append('./output')
sys.path.append('./dataset')

import blitnet as bn
import numpy as np
import torch.nn as nn
import torch.quantization as quantization

from loggers import model_logger_quant
from VPRTempoQuantTrain import generate_model_name_quant
from dataset import CustomImageDataset, ProcessImage
from torch.utils.data import DataLoader
from torch.ao.quantization import QuantStub, DeQuantStub
from tqdm import tqdm
from prettytable import PrettyTable
from metrics import recallAtK

#from main import parse_network

class VPRTempoQuant(nn.Module):
    def __init__(self, args, dims):
        super(VPRTempoQuant, self).__init__()

        # Set the arguments
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
        setattr(self, 'dims', dims)
        # Set the dataset file
        self.dataset_file = os.path.join('./dataset', self.dataset + '.csv')

        # Set the model logger and return the device
        self.device = model_logger_quant(self)

        # Add quantization stubs for Quantization Aware Training (QAT)
        self.quant = QuantStub()
        self.dequant = DeQuantStub() 

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        # Define layer architecture
        self.input = int(self.dims[0]*self.dims[1])
        self.feature = int(self.input * 2)
        self.output = int(args.num_places / args.num_modules)

        """
        Define trainable layers here
        """
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature],
            device=self.device,
            inference=True
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            device=self.device,
            inference=True
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

    def evaluate(self, model, test_loader):
        """
        Run the inferencing model and calculate the accuracy.

        :param test_loader: Testing data loader
        :param layers: Layers to pass data through
        """
        # Determine the Hardtahn max value
        maxSpike = (1//model.quant.scale).item()
        # Define the sequential inference model
        self.inference = nn.Sequential(
            self.feature_layer.w,
            nn.Hardtanh(0, maxSpike),
            nn.ReLU(),
            self.output_layer.w,
            nn.Hardtanh(0, maxSpike),
            nn.ReLU()
        )
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.num_places,
                    desc="Running the test network",
                    position=0)
        # Initiliaze the output spikes variable
        out = []
        # Run inference for the specified number of timesteps
        for spikes, labels in test_loader:
            # Set device
            spikes, labels = spikes.to(self.device), labels.to(self.device)
            # Pass through previous layers if they exist
            spikes = self.forward(spikes)
            # Add output spikes to list
            out.append(spikes.detach().cpu().tolist())
            pbar.update(1)

        # Close the tqdm progress bar
        pbar.close()

        # Rehsape output spikes into a similarity matrix
        out = np.reshape(np.array(out),(self.num_places,self.num_places))
        # Calculate and print the Recall@N
        N = [1,5,10,15,20,25]
        R = []
        # Create GT matrix
        GT = np.zeros((self.num_places,self.num_places), dtype=int)
        for n in range(len(GT)):
            GT[n,n] = 1
        for n in N:
            R.append(recallAtK(out,GThard=GT,K=n))
        # Print the results
        table = PrettyTable()
        table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
        model.logger.info(table)

    def forward(self, spikes):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        
        spikes = self.quant(spikes)
        spikes = self.inference(spikes)
        spikes = self.dequant(spikes)
        
        return spikes
        
    def load_model(self, model_path):
        """
        Load pre-trained model and set the state dictionary keys.
        """
        self.load_state_dict(torch.load(model_path, map_location=self.device),
                             strict=False)
            
def check_pretrained_model(model_name):
    """
    Check if a pre-trained model exists and tell user if it does not.
    """
    if not os.path.exists(os.path.join('./models', model_name)):
        model.logger.info("A pre-trained network does not exist: please train one using VPRTempoQuant_Trainer")
        pretrain = 'n'
    else:
        pretrain = 'y'
    return pretrain

def run_inference_quant(model, model_name, qconfig):
    """
    Run inference on a pre-trained model.

    :param model: Model to run inference on
    :param model_name: Name of the model to load
    :param qconfig: Quantization configuration
    """
    # Initialize the image transforms and datasets
    image_transform = ProcessImage(model.dims, model.patches)
    test_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                    base_dir=model.data_dir,
                                    img_dirs=model.query_dir,
                                    transform=image_transform,
                                    skip=model.filter,
                                    max_samples=model.num_places)
    # Initialize the data loader
    test_loader = DataLoader(test_dataset, 
                             batch_size=1, 
                             shuffle=False,
                             num_workers=8,
                             persistent_workers=True)
    # Set the model to evaluation mode and set configuration
    model.eval()
    model.qconfig = qconfig

    # Apply quantization configurations to all layers in layer_dict
    for layer_name, _ in model.layer_dict.items():
        getattr(model, layer_name).qconfig = qconfig
    # Prepare and convert the model to a quantized model
    model = quantization.prepare(model, inplace=False)
    model = quantization.convert(model, inplace=False)
    # Load the model
    model.load_model(os.path.join('./models', model_name))

    # Use evaluate method for inference accuracy
    model.evaluate(model, test_loader)