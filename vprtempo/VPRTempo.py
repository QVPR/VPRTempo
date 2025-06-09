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

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import vprtempo.src.blitnet as bn

from tqdm import tqdm
from vprtempo.src.demo import demo
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from vprtempo.src.download import get_data_model
from vprtempo.src.metrics import recallAtK, createPR
from vprtempo.src.dataset import CustomImageDataset, ProcessImage

class VPRTempo(nn.Module):
    def __init__(self, args, dims, logger, num_modules, output_folder, out_dim, out_dim_remainder=None):
        super(VPRTempo, self).__init__()

        # Set the args
        if args is not None:
            self.args = args
            for arg in vars(args):
                setattr(self, arg, getattr(args, arg))
        setattr(self, 'dims', dims)

        # Set the device
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Set input args
        self.logger = logger
        self.num_modules = num_modules
        self.output_folder = output_folder

        # Set the dataset file
        self.dataset_file = os.path.join('./vprtempo/dataset', f'{self.dataset}-{self.query_dir}' + '.csv')  
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

        # set model name for default demo
        self.demo = './vprtempo/models/springfall_VPRTempo_IN3136_FN6272_DB500.pth'

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

    def evaluate(self, models, test_loader):
        """
        Run the inferencing model and calculate the accuracy.

        :param models: Models to run inference on, each model is a VPRTempo module
        :param test_loader: Testing data loader
        """
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.query_places,
                    desc="Running the test network",
                    position=0)
        self.inferences = []
        for model in models:
            self.inferences.append(nn.Sequential(
                model.feature_layer.w,
                model.output_layer.w,
            ))
            self.inferences[-1].to(torch.device(self.device))
        # Initiliaze the output spikes variable
        out = []
        labels = []

        # Run inference for the specified number of timesteps
        for spikes, label in test_loader:
            # Set device
            spikes = spikes.to(self.device)
            labels.append(label.detach().cpu().item())
            # Forward pass
            spikes = self.forward(spikes)
            # Add output spikes to list
            out.append(spikes.detach().cpu())
            pbar.update(1)

        # Close the tqdm progress bar
        pbar.close()
        # Rehsape output spikes into a similarity matrix
        out = torch.stack(out, dim=2)
        out = out.squeeze(0).numpy()
       
        if self.skip != 0:
            GT = np.zeros((model.database_places, model.query_places))
            skip = model.skip // model.filter
            # Create an array of indices for the query dimension
            query_indices = np.arange(model.query_places)

            # Set the ones on the diagonal starting at row `skip`
            GT[skip + query_indices, query_indices] = 1
        else:
            GT = np.eye(model.database_places, model.query_places)

        # Apply GT tolerance
        if self.GT_tolerance > 0:
            # Get the number of rows and columns
            num_rows, num_cols = GT.shape
            
            # Iterate over each column
            for col in range(num_cols):
                # Find the indices of rows where GT has a 1 in the current column
                ones_indices = np.where(GT[:, col] == 1)[0]
                
                # For each index with a 1, set 1s in GTtol within the specified vertical distance
                for row in ones_indices:
                    # Determine the start and end rows, ensuring they are within bounds
                    start_row = max(row - self.GT_tolerance, 0)
                    end_row = min(row + self.GT_tolerance + 1, num_rows)  # +1 because upper bound is exclusive
                    
                    # Set the range in GTtol to 1
                    GT[start_row:end_row, col] = 1
        
        # If user specified, generate a PR curve
        if model.PR_curve:
            # Create PR curve
            P, R = createPR(out, GT, matching='single', n_thresh=100)
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
            if not model.demo:
                plt.plot(R,P)    
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.show()

                plt.close()

        if model.sim_mat and not model.demo:
            # Create a figure and a set of subplots
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            # Plot each matrix using matshow
            cax1 = axs[0].matshow(out, cmap='viridis')
            fig.colorbar(cax1, ax=axs[0], shrink=0.8)
            axs[0].set_title('Similarity matrix')

            cax2 = axs[1].matshow(GT, cmap='plasma')
            fig.colorbar(cax2, ax=axs[1], shrink=0.8)
            axs[1].set_title('GT')

            # Adjust layout
            plt.tight_layout()
            plt.show()

            plt.close()
        

        # Recall@N
        N = [1,5,10,15,20,25] # N values to calculate
        RN = [] # Recall@N values
        # Calculate Recall@N
        for n in N:
            RN.append(round(recallAtK(out, GT, K=n),2))

        if model.demo:
            # Run the demo with the output spikes and ground truth
            demo(model.data_dir, model.query_dir[0], model.database_dirs[0], out, GT, N, RN, R, P)

            return

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
        # Initialize the output spikes tensor
        in_spikes = spikes.detach().clone()
        outputs = []  # List to collect output tensors

        # Run inferencing across modules
        for inference in self.inferences:
            out_spikes = inference(in_spikes)
            outputs.append(out_spikes)  # Append the output tensor to the list

        # Concatenate along the desired dimension
        concatenated_output = torch.cat(outputs, dim=1)
        
        return concatenated_output
        
    def load_model(self, models, model_path):
        """
        Load pre-trained model and set the state dictionary keys.
        """
        # check if model exists, download the demo data and model if defaults are set
        if not os.path.exists(model_path):
            if model_path == self.demo:
                get_data_model()
            else:
                raise ValueError(f"Model path {model_path} does not exist.")
            
        combined_state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        for i, model in enumerate(models):  # models_classes is a list of model classes
            model.load_state_dict(combined_state_dict[f'model_{i}'])
            model.eval()  # Set the model to inference mode

def run_inference(models, model_name):
    """
    Run inference on a pre-trained model.

    :param models: Models to run inference on, each model is a VPRTempo module
    :param model_name: Name of the model to load
    """
    # Set first index model as the main model for parameters
    model = models[0]
    # Initialize the image transforms
    image_transform = ProcessImage(model.dims, model.patches)
    
    # Initialize the test dataset
    test_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                      base_dir=model.data_dir,
                                      img_dirs=model.query_dir,
                                      transform=image_transform,
                                      max_samples=model.query_places,
                                      filter=model.filter,
                                      skip=model.skip)

    # Initialize the data loader
    test_loader = DataLoader(test_dataset, 
                             batch_size=1,
                             num_workers=4,
                             persistent_workers=True)

    # Load the model
    model.load_model(models, os.path.join('./vprtempo/models', model_name))

    # Use evaluate method for inference accuracy
    with torch.no_grad():
        model.evaluate(models, test_loader)