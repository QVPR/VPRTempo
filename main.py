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
import argparse
import sys
sys.path.append('./src')
sys.path.append('./networks/base')
sys.path.append('./networks/quantized')

import torch.quantization as quantization

from VPRTempoTrain import VPRTempoTrain, generate_model_name, check_pretrained_model, train_new_model
from VPRTempo import VPRTempo, run_inference
from VPRTempoQuantTrain import VPRTempoQuantTrain, generate_model_name_quant, train_new_model_quant
from VPRTempoQuant import VPRTempoQuant, run_inference_quant
from loggers import model_logger, model_logger_quant

def initialize_and_run_model(args,dims):
    # If user wants to train a new network
    if args.train_new_model:
        # If using quantization aware training
        if args.quantize:
            models = []
            logger = model_logger_quant()
            # Get the quantization config
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            for _ in range(args.num_modules):
                # Initialize the model
                model = VPRTempoQuantTrain(args, dims, logger)
                model.train()
                model.qconfig = qconfig
                models.append(model)
            # Generate the model name
            model_name = generate_model_name_quant(model)
            # Check if the model has been trained before
            check_pretrained_model(model_name)
            # Train the model
            train_new_model_quant(models, model_name, qconfig)
        else: # Normal model
            models = []
            logger = model_logger()
            for _ in range(args.num_modules):
                # Initialize the model
                model = VPRTempoTrain(args, dims, logger)
                models.append(model)
            # Generate the model name
            model_name = generate_model_name(model)
            # Check if the model has been trained before
            check_pretrained_model(model_name)
            # Train the model
            train_new_model(models, model_name)
    # Run the inference network
    else:
        # Set the quantization configuration
        if args.quantize:
            models = []
            logger = model_logger_quant()
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            for _ in range(args.num_modules):
                # Initialize the model
                model = VPRTempoQuant(dims, args, logger)
                model.eval()
                model.qconfig = qconfig
                model = quantization.prepare(model, inplace=False)
                model = quantization.convert(model, inplace=False)
                models.append(model)
            # Generate the model name
            model_name = generate_model_name_quant(model)
            # Run the quantized inference model
            run_inference_quant(models, model_name, qconfig)
        else:
            models = []
            logger = model_logger()
            for _ in range(args.num_modules):
                # Initialize the model
                model = VPRTempo(dims, args, logger)
                models.append(model)
            # Generate the model name
            model_name = generate_model_name(model)
            # Run the inference model
            run_inference(models, model_name)

def parse_network(use_quantize=False, train_new_model=False):
    '''
    Define the base parameter parser (configurable by the user)
    '''
    parser = argparse.ArgumentParser(description="Args for base configuration file")

    # Define the dataset arguments
    parser.add_argument('--dataset', type=str, default='nordland',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='./dataset/',
                            help="Directory where dataset files are stored")
    parser.add_argument('--num_places', type=int, default=500,
                            help="Number of places to use for training and/or inferencing")
    parser.add_argument('--num_modules', type=int, default=1,
                            help="Number of expert modules to use split images into")
    parser.add_argument('--max_module', type=int, default=500,
                            help="Maximum number of images per module")
    parser.add_argument('--database_dirs', nargs='+', default=['spring', 'fall'],
                            help="Directories to use for training")
    parser.add_argument('--query_dir', nargs='+', default=['summer'],
                            help="Directories to use for testing")

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
    
    # If the function is called with specific arguments, override sys.argv
    if use_quantize or train_new_model:
        sys.argv = ['']
        if use_quantize:
            sys.argv.append('--quantize')
        if train_new_model:
            sys.argv.append('--train_new_model')

    # Output base configuration
    args = parser.parse_args()
    dims = [int(x) for x in args.dims.split(",")]

    # Run the network with the desired settings
    initialize_and_run_model(args,dims)

if __name__ == "__main__":
    # User input to determine if using quantized network or to train new model 
    parse_network(use_quantize=False, 
                  train_new_model=False)