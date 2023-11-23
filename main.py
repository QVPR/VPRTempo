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
import json

import torch.quantization as quantization

from VPRTempoTrain import VPRTempoTrain, generate_model_name, check_pretrained_model, train_new_model
from VPRTempo import VPRTempo, run_inference
from VPRTempoQuantTrain import VPRTempoQuantTrain, generate_model_name_quant, train_new_model_quant
from VPRTempoQuant import VPRTempoQuant, run_inference_quant

def initialize_and_run_model(train: bool, quantize: bool):
    if train:
        if quantize:
            # Initialize the quantized model
            model = VPRTempoQuantTrain()
            # Get the quantization config
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            # Generate the model name
            model_name = generate_model_name_quant(model)
            # Check if the model has been trained before
            check_pretrained_model(model_name)
            # Train the model
            train_new_model_quant(model, model_name, qconfig)
        else:
            # Initialize the model
            model = VPRTempoTrain()
            # Generate the model name
            model_name = generate_model_name(model)
            # Check if the model has been trained before
            check_pretrained_model(model_name)
            # Train the model
            train_new_model(model, model_name)
    else:
        # Set the quantization configuration
        if quantize:
            # Initialize the quantized model
            model = VPRTempoQuant()
            # Get the quantization config
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            # Generate the model name
            model_name = generate_model_name_quant(model)
            # Run the quantized inference model
            run_inference_quant(model, model_name, qconfig)
        else:
            # Initialize the model
            model = VPRTempo()
            # Generate the model name
            model_name = generate_model_name(model)
            # Run the inference model
            run_inference(model, model_name)

def main():
    '''
    Define the base parameter parser (configurable by the user)
    '''
    base_parser = argparse.ArgumentParser(description="Args for base configuration file")

    # Define the dataset arguments
    base_parser.add_argument('--dataset', type=str, default='nordland',
                            help="Dataset to use for training and/or inferencing")
    base_parser.add_argument('--data_dir', type=str, default='./dataset/',
                            help="Directory where dataset files are stored")
    base_parser.add_argument('--num_places', type=int, default=500,
                            help="Number of places to use for training and/or inferencing")
    base_parser.add_argument('--num_modules', type=int, default=1,
                            help="Number of expert modules to use split images into")
    base_parser.add_argument('--database_dirs', nargs='+', default=['spring', 'fall'],
                            help="Directories to use for training")
    base_parser.add_argument('--query_dir', nargs='+', default=['summer'],
                            help="Directories to use for testing")

    # Define training parameters
    base_parser.add_argument('--filter', type=int, default=8,
                            help="Images to skip for training and/or inferencing")
    base_parser.add_argument('--epoch', type=int, default=4,
                            help="Number of epochs to train the model")

    # Define image transformation parameters
    base_parser.add_argument('--patches', type=int, default=15,
                            help="Number of patches to generate for patch normalization image into")
    base_parser.add_argument('--dims', nargs='+', type=int, default=[56,56],
                            help="Dimensions to resize the image to")
    
    # Output base configuration
    base_args = base_parser.parse_args()

    # Write to base_config JSON file
    with open('./config/base_config.json', 'w') as file:
        json.dump(vars(base_args), file, indent=4)

    '''
    Define network architecture parameter parser
    '''
    network_parser = argparse.ArgumentParser(description="Args for network architecture configuration file")

    # Define network architecure (number of neurons in each layer)
    network_parser.add_argument('--input', type=int, default=base_args.dims[0]*base_args.dims[1],
                            help="Number of input neurons")
    network_parser.add_argument('--feature', type=int, default=(base_args.dims[0]*base_args.dims[1])*2,
                            help="Number of feature neurons")
    network_parser.add_argument('--output', type=int, default=int(base_args.num_places/base_args.num_modules),
                            help="Number of output neurons")
    
    # Determine total number of timesteps
    network_parser.add_argument('--T', type=int, 
            default=(base_args.num_places / base_args.num_modules) * len(base_args.database_dirs) * base_args.epoch)

    # Determine network functionality
    parser = argparse.ArgumentParser(description="Determine training or inferencing and the quantization scheme")
    parser.add_argument('--train', action='store_true',
                            help="Flag to run the training or inferencing model")
    parser.add_argument('--quantize', action='store_true',
                            help="Enable/disable quantization for the model")
    args = parser.parse_args()
    initialize_and_run_model(args.train,
                             args.quantize)

if __name__ == "__main__":
    main()
