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
import json
import torch
import random

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import vprtempo.src.blitnet as bn

from tqdm import tqdm
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Subset
from vprtempo.src.metrics import recallAtK, createPR
from torch.ao.quantization import QuantStub, DeQuantStub
from vprtempo.src.dataset import CustomImageDataset, ProcessImage

#from main import parse_network

class VPRTempoQuant(nn.Module):
    def __init__(self, args, dims, logger, num_modules, output_folder, out_dim, out_dim_remainder=None):
        super(VPRTempoQuant, self).__init__()

        # Set the args
        if args is not None:
            self.args = args
            for arg in vars(args):
                setattr(self, arg, getattr(args, arg))
        setattr(self, 'dims', dims)

        # Set the device
        self.device = "cpu"

        # Set input args
        self.logger = logger
        self.num_modules = num_modules
        self.output_folder = output_folder

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Set the dataset file
        self.dataset_file = os.path.join('./vprtempo/dataset', self.dataset + '.csv')  
        self.query_dir = [dir.strip() for dir in self.query_dir.split(',')]

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0
        self.database_dirs = [dir.strip() for dir in self.database_dirs.split(',')]

        # Define layer architecture
        self.input = int(self.dims[0]*self.dims[1])
        self.feature = int(self.input * 2)

        # Output dimension changes for final module if not an even distribution of places
        if not out_dim_remainder is None:
            self.output = out_dim_remainder
        else:
            self.output = out_dim

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

    def evaluate(self, models, test_loader, layers=None):
        """
        Run the inferencing model and calculate the accuracy.

        :param test_loader: Testing data loader
        :param layers: Layers to pass data through
        """
        # Determine the Hardtahn max value
        maxSpike = (1//models[0].quant.scale).item()
        # Define the sequential inference model
        self.inferences = []
        for model in models:
            self.inferences.append(nn.Sequential(
                model.feature_layer.w,
                nn.Hardtanh(0, maxSpike),
                nn.ReLU(),
                model.output_layer.w,
                nn.Hardtanh(0, maxSpike),
                nn.ReLU()
            ))
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.query_places,
                    desc="Running the test network",
                    position=0)
        # Initiliaze the output spikes variable
        out = []
        labels = []
        # Run inference for the specified number of timesteps
        for spikes, label in test_loader:
            # Set device
            spikes = spikes.to(self.device)
            labels.append(label.detach().item())
            # Pass through previous layers if they exist
            spikes = self.forward(spikes)
            # Add output spikes to list
            out.append(spikes.detach().cpu().tolist())
            pbar.update(1)

        # Close the tqdm progress bar
        pbar.close()

        # Rehsape output spikes into a similarity matrix
        out = np.reshape(np.array(out),(model.query_places,model.database_places))

        # Create GT matrix
        GT = np.zeros((model.query_places,model.database_places), dtype=int)
        for n, ndx in enumerate(labels):
            if model.filter !=1:
                ndx = ndx//model.filter
            GT[n,ndx] = 1

        # Create GT soft matrix
        if model.GT_tolerance > 0:
            GTsoft = np.zeros((model.query_places,model.database_places), dtype=int)
            for n, ndx in enumerate(labels):
                if model.filter !=1:
                    ndx = ndx//model.filter
                GTsoft[n, ndx] = 1
                # Apply tolerance
                for i in range(max(0, n - model.GT_tolerance), min(model.query_places, n + model.GT_tolerance + 1)):
                    GTsoft[i, ndx] = 1
        else:
            GTsoft = None
        
        # If user specified, generate a PR curve
        if model.PR_curve:
            # Create PR curve
            P, R = createPR(out, GThard=GT, GTsoft=GTsoft, matching='single', n_thresh=100)
            # Combine P and R into a list of lists
            PR_data = {
                    "Precision": P,
                    "Recall": R
                }
            output_file = "PR_curve_data.json"
            # Construct the full path
            full_path = f"{model.output_folder}/{output_file}"
            # Write the data to a JSON file
            with open(full_path, 'w') as file:
                json.dump(PR_data, file) 
            # Plot PR curve
            plt.plot(R,P)    
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.show()

        if model.sim_mat:
            # Create a figure and a set of subplots
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Plot each matrix using matshow
            cax1 = axs[0].matshow(out, cmap='viridis')
            fig.colorbar(cax1, ax=axs[0], shrink=0.8)
            axs[0].set_title('Similarity matrix')

            cax2 = axs[1].matshow(GT, cmap='plasma')
            fig.colorbar(cax2, ax=axs[1], shrink=0.8)
            axs[1].set_title('GT')

            cax3 = axs[2].matshow(GTsoft, cmap='inferno')
            fig.colorbar(cax3, ax=axs[2], shrink=0.8)
            axs[2].set_title('GT-soft')

            # Adjust layout
            plt.tight_layout()
            plt.show()

        # Recall@N
        N = [1,5,10,15,20,25] # N values to calculate
        R = [] # Recall@N values
        # Calculate Recall@N
        for n in N:
            R.append(round(recallAtK(out,GThard=GT,GTsoft=GTsoft,K=n),2))
        # Print the results
        table = PrettyTable()
        table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
        self.logger.info(table)

    def forward(self, spikes):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        spikes = self.quant(spikes)
        in_spikes = spikes.detach().clone()
        outputs = []  # List to collect output tensors

        for inference in self.inferences:
            out_spikes = inference(in_spikes)
            outputs.append(out_spikes)  # Append the output tensor to the list

        # Concatenate along the desired dimension
        concatenated_output = torch.cat(outputs, dim=1)
        spikes = self.dequant(concatenated_output)

        return spikes
        
    def load_model(self, models, model_path):
        """
        Load pre-trained model and set the state dictionary keys.
        """
        combined_state_dict = torch.load(model_path, map_location=self.device)

        for i, model in enumerate(models):  # models_classes is a list of model classes

            model.load_state_dict(combined_state_dict[f'model_{i}'])
            model.eval()  # Set the model to inference mode

def run_inference_quant(models, model_name):
    """
    Run inference on a pre-trained model.

    :param models: Models to run inference on, each model is a VPRTempo module
    :param model_name: Name of the model to load
    """
    # Set first index model as the main model for parameters
    model = models[0]
    # Initialize the image transforms
    image_transform = ProcessImage(model.dims, model.patches)

    # Determines if querying a subset of the database or the entire database
    if model.query_places == model.database_places:
        subset = False # Entire database
    elif model.query_places < model.database_places:  
        subset = True # Subset of the database
    else:
        raise ValueError("The number of query places must be less than or equal to the number of database places.")
    
    # Initialize the test dataset
    test_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                      base_dir=model.data_dir,
                                      img_dirs=model.query_dir,
                                      transform=image_transform,
                                      max_samples=model.database_places,
                                      skip=model.filter)
    
    # If using a subset of the database
    if subset:
        if model.shuffle: # For a randomized selection of database places
             test_dataset = Subset(test_dataset, random.sample(range(len(test_dataset)), model.query_places))
        else: # For a sequential selection of database places
            indices = [i for i in range(model.database_places) if i % model.filter == 0]
            # Limit to the desired number of queries
            indices = indices[:model.query_places]
            # Create the subset
            test_dataset = Subset(test_dataset, indices)


    # Initialize the data loader
    test_loader = DataLoader(test_dataset, 
                             batch_size=1, 
                             shuffle=model.shuffle,
                             num_workers=8,
                             persistent_workers=True)

    # Load the model
    model.load_model(models, os.path.join('./vprtempo/models', model_name))

    # Use evaluate method for inference accuracy
    with torch.no_grad():
        model.evaluate(models, test_loader)