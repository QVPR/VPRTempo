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
import sys
import torch
import argparse

import torch.quantization as quantization

from tqdm import tqdm
from vprtempo.VPRTempo import VPRTempo, run_inference
from vprtempo.VPRTempoTrain import VPRTempoTrain, train_new_model
from vprtempo.src.loggers import model_logger, model_logger_quant
from vprtempo.VPRTempoQuant import VPRTempoQuant, run_inference_quant
from vprtempo.VPRTempoQuantTrain import VPRTempoQuantTrain, train_new_model_quant

def generate_model_name(model,quant=False):
    """
    Generate the model name based on its parameters.
    """
    if quant:
        model_name = (''.join(model.database_dirs)+"_"+
                "VPRTempoQuant_" +
                "IN"+str(model.input)+"_" +
                "FN"+str(model.feature)+"_" + 
                "DB"+str(model.database_places) +
                ".pth")
    else:
        model_name = (''.join(model.database_dirs)+"_"+
                "VPRTempo_" +
                "IN"+str(model.input)+"_" +
                "FN"+str(model.feature)+"_" + 
                "DB"+str(model.database_places) +
                ".pth")
    return model_name

def check_pretrained_model(model_name):
    """
    Check if a pre-trained model exists and prompt the user to retrain if desired.
    """
    if os.path.exists(os.path.join('./vprtempo/models', model_name)):
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        retrain = input(prompt).strip().lower()
        if retrain == 'y':
            return True
        elif retrain == 'n':
            print('Training new model cancelled')
            sys.exit()

def initialize_and_run_model(args,dims):
    """
    Run the VPRTempo/VPRTempoQuant training or inference models.
    
    :param args: Arguments set for the network
    :param dims: Dimensions of the network
    """
    # Determine number of modules to generate based on user input
    places = args.database_places # Copy out number of database places

    # Caclulate number of modules
    num_modules = 1
    while places > args.max_module:
        places -= args.max_module
        num_modules += 1

    # If the final module has less than max_module, reduce the dim of the output layer
    remainder = args.database_places % args.max_module
    if remainder != 0: # There are remainders, adjust output neuron count in final module
        out_dim = int((args.database_places - remainder) / (num_modules - 1))
        final_out_dim = remainder
    else: # No remainders, all modules are even
        out_dim = int(args.database_places / num_modules)
        final_out_dim = out_dim

    # If user wants to train a new network
    if args.train_new_model:
        # If using quantization aware training
        if args.quantize:
            models = []
            logger = model_logger_quant() # Initialize the logger
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            # Create the modules    
            final_out = None
            for mod in tqdm(range(num_modules), desc="Initializing modules"):
                model = VPRTempoQuantTrain(args, dims, logger, num_modules, out_dim, out_dim_remainder=final_out) # Initialize the model
                model.train()
                model.qconfig = qconfig
                quantization.prepare_qat(model, inplace=True)
                models.append(model) # Create module list
                if mod == num_modules - 2:
                    final_out = final_out_dim
            # Generate the model name
            model_name = generate_model_name(model,args.quantize)
            # Check if the model has been trained before
            check_pretrained_model(model_name)
            # Get the quantization config
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            # Train the model
            train_new_model_quant(models, model_name)

        # Base model    
        else:
            models = []
            logger = model_logger() # Initialize the logger

            # Create the modules    
            final_out = None
            for mod in tqdm(range(num_modules), desc="Initializing modules"):
                model = VPRTempoTrain(args, dims, logger, num_modules, out_dim, out_dim_remainder=final_out) # Initialize the model
                model.to(torch.device('cpu')) # Move module to CPU for storage (necessary for large models)
                models.append(model) # Create module list
                if mod == num_modules - 2:
                    final_out = final_out_dim

            # Generate the model name
            model_name = generate_model_name(model)
            print(f"Model name: {model_name}")
            # Check if the model has been trained before
            check_pretrained_model(model_name)
            # Train the model
            train_new_model(models, model_name)

    # Run the inference network
    else:
        # Set the quantization configuration
        if args.quantize:
            models = []
            logger, output_folder = model_logger_quant()
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            final_out = None
            for _ in tqdm(range(num_modules), desc="Initializing modules"):
                # Initialize the model
                model = VPRTempoQuant(
                    args,
                    dims,
                    logger,
                    num_modules,
                    output_folder,
                    out_dim,
                    out_dim_remainder=final_out
                    ) 
                model.eval()
                model.qconfig = qconfig
                quantization.prepare(model, inplace=True)
                quantization.convert(model, inplace=True)
                models.append(model)
            # Generate the model name
            model_name = generate_model_name(model, args.quantize)
            # Run the quantized inference model
            run_inference_quant(models, model_name)
        else:
            models = []
            logger, output_folder = model_logger() # Initialize the logger
            places = args.database_places # Copy out number of database places

            # Create the modules    
            final_out = None
            for mod in tqdm(range(num_modules), desc="Initializing modules"):
                model = VPRTempo(
                    args,
                    dims,
                    logger,
                    num_modules,
                    output_folder,
                    out_dim,
                    out_dim_remainder=final_out
                    ) 
                model.eval()
                model.to(torch.device('cpu')) # Move module to CPU for storage (necessary for large models)
                models.append(model) # Create module list
                if mod == num_modules - 2:
                    final_out = final_out_dim
            # Generate the model name
            model_name = generate_model_name(model)
            print(f"Model name: {model_name}")
            # Run the inference model
            run_inference(models, model_name)

def parse_network():
    '''
    Define the base parameter parser (configurable by the user)
    '''
    parser = argparse.ArgumentParser(description="Args for base configuration file")

    # Define the dataset arguments
    parser.add_argument('--dataset', type=str, default='nordland',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='./vprtempo/dataset/',
                            help="Directory where dataset files are stored")
    parser.add_argument('--database_places', type=int, default=500,
                            help="Number of places to use for training")
    parser.add_argument('--query_places', type=int, default=500,
                            help="Number of places to use for inferencing")
    parser.add_argument('--max_module', type=int, default=500,
                            help="Maximum number of images per module")
    parser.add_argument('--database_dirs', type=str, default='spring,fall',
                            help="Directories to use for training")
    parser.add_argument('--query_dir', type=str, default='summer',
                            help="Directories to use for testing")
    parser.add_argument('--GT_tolerance', type=int, default=0,
                            help="Ground truth tolerance for matching")
    parser.add_argument('--skip', type=int, default=0,
                            help="Images to skip for training and/or inferencing")

    # Define training parameters
    parser.add_argument('--filter', type=int, default=8,
                            help="Images to skip for training and/or inferencing")
    parser.add_argument('--epoch', type=int, default=4,
                            help="Number of epochs to train the model")

    # Define image transformation parameters
    parser.add_argument('--patches', type=int, default=15,
                            help="Number of patches to generate for patch normalization image into")
    parser.add_argument('--dims', type=str, default="56,56",
                        help="Dimensions to resize the image to")

    # Define the network functionality
    parser.add_argument('--train_new_model', action='store_true',
                            help="Flag to run the training or inferencing model")
    parser.add_argument('--quantize', action='store_true',
                            help="Enable/disable quantization for the model")
    
    # Define metrics functionality
    parser.add_argument('--PR_curve', action='store_true',
                            help="Flag to generate a Precision-Recall curve")
    parser.add_argument('--sim_mat', action='store_true',
                            help="Flag to plot the similarity matrix, GT, and GTsoft")
    
    # Run the demo
    parser.add_argument('--demo', action='store_true',
                            help="Flag to run the demo script")
    
    # Output base configuration
    args = parser.parse_args()
    dims = [int(x) for x in args.dims.split(",")]

    # Run the network with the desired settings
    initialize_and_run_model(args,dims)

if __name__ == "__main__":
    parse_network()