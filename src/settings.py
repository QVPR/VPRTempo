import os
import torch
import logging
import json
import sys
sys.path.append('./config')

from datetime import datetime

def configure(model):
    """
    Configure the model settings
    """
    base_config = {
        "dataset": "nordland",
        "number_places": 500,
        "number_modules": 1,
        "database_dirs": ["spring", "fall"],
        "query_dir": ["summer"],
        "filter": 8,
        "epoch": 4,
        "patches": 15,
        "dims": [56,56],
    }

    # Write to base_config JSON file
    with open('./config/base_config.json', 'w') as file:
        json.dump(base_config, file, indent=4)
    
    # Set default paths if the provided paths are not valid directories
    if not os.path.isdir(getattr(model, 'trainingPath', '')):
        model.trainingPath = '../dataset/'

    if not os.path.isdir(getattr(model, 'testPath', '')):
        model.testPath = '../dataset/'

    # Now, check if the dataset_file exists based on the determined paths
    if not os.path.exists(os.path.join('./dataset', model.dataset + '.csv')):
        model.dataset_file = os.path.join('../dataset', model.dataset + '.csv')
    else:
        model.dataset_file = os.path.join('./dataset', model.dataset + '.csv')

    # Now, check the conditions using assert statements
    assert (len(model.dataset) != 0), "Dataset not defined, see README.md for details on setting up images"
    assert (os.path.isdir(model.trainingPath)), "Training path not set or path does not exist, specify for model.trainingPath"
    assert (os.path.isdir(model.testPath)), "Test path not set or path does not exist, specify for model.testPath"
    assert (os.path.isdir(model.trainingPath + model.locations[0])), "Images must be organized into folders based on locations, see README.md for details"
    assert (os.path.isdir(model.testPath + model.test_locations[0])), "Images must be organized into folders based on locations, see README.md for details"

    # Output the training and testing directories
    model.training_dirs = []
    for n in model.locations:
        model.training_dirs.append(os.path.join(model.trainingPath,n))    
    model.testing_dirs = []
    for n in model.test_locations:
        model.testing_dirs.append(os.path.join(model.testPath,n))

    model.location_repeat = len(model.locations) # Number of times to repeat the locations

    """
    These parameters are used to define the network architecture
    """
    
    # Set the torch device
    if not model.quantize:
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        model.device = torch.device("cpu")
    if model.device.type == "cuda":
        torch.cuda.init()
        torch.cuda.synchronize(device=model.device)

    # Determine the total number of timesteps across training images, modules, and location repeats
    model.T = int((model.number_training_images / model.number_modules) * model.location_repeat) * model.epoch

def model_logger(model): 
    """
    Configure the model logger
    """   
    if os.path.isdir('../output'):
        now = datetime.now()
        model.output_folder = '../output/' + now.strftime("%d%m%y-%H-%M-%S")
    else:
        now = datetime.now()
        model.output_folder = './output/' + now.strftime("%d%m%y-%H-%M-%S")
    
    os.mkdir(model.output_folder)
    # Create the logger
    model.logger = logging.getLogger("VPRTempo")
    if (model.logger.hasHandlers()):
        model.logger.handlers.clear()
    # Set the logger level
    model.logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename=model.output_folder + "/logfile.log",
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    # Add the logger to the console (if specified)
    if model.log:
        model.logger.addHandler(logging.StreamHandler())
        
    model.logger.info('')
    model.logger.info('██╗   ██╗██████╗ ██████╗ ████████╗███████╗███╗   ███╗██████╗  ██████╗') 
    model.logger.info('██║   ██║██╔══██╗██╔══██╗╚══██╔══╝██╔════╝████╗ ████║██╔══██╗██╔═══██╗')
    model.logger.info('██║   ██║██████╔╝██████╔╝   ██║   █████╗  ██╔████╔██║██████╔╝██║   ██║')
    model.logger.info('╚██╗ ██╔╝██╔═══╝ ██╔══██╗   ██║   ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██║   ██║')
    model.logger.info(' ╚████╔╝ ██║     ██║  ██║   ██║   ███████╗██║ ╚═╝ ██║██║     ╚██████╔╝')
    model.logger.info('  ╚═══╝  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝      ╚═════╝ ')
    model.logger.info('-----------------------------------------------------------------------')
    model.logger.info('Temporally Encoded Spiking Neural Network for Visual Place Recognition v1.1.0')
    model.logger.info('Queensland University of Technology, Centre for Robotics')
    model.logger.info('')
    model.logger.info('© 2023 Adam D Hines, Peter G Stratton, Michael Milford, Tobias Fischer')
    model.logger.info('MIT license - https://github.com/QVPR/VPRTempo')
    model.logger.info('\\\\\\\\\\\\\\\\\\\\\\\\')
    model.logger.info('')
    if model.quantize:
        model.logger.info('Quantization enabled')
        model.logger.info('Current device is: CPU')
    else:
        if torch.cuda.is_available():
            model.logger.info('CUDA available: ' + str(torch.cuda.is_available()))
            current_device = torch.cuda.current_device()
            model.logger.info('Current device is: ' + str(torch.cuda.get_device_name(current_device)))
        else:
            model.logger.info('CUDA available: ' + str(torch.cuda.is_available()))
            model.logger.info('Current device is: CPU')
    model.logger.info('')

def model_logger_quant(model): 
    """
    Configure the model logger
    """   
    if os.path.isdir('../output'):
        now = datetime.now()
        model.output_folder = '../output/' + now.strftime("%d%m%y-%H-%M-%S")
    else:
        now = datetime.now()
        model.output_folder = './output/' + now.strftime("%d%m%y-%H-%M-%S")
    
    os.mkdir(model.output_folder)
    # Create the logger
    model.logger = logging.getLogger("VPRTempo")
    if (model.logger.hasHandlers()):
        model.logger.handlers.clear()
    # Set the logger level
    model.logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename=model.output_folder + "/logfile.log",
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    # Add the logger to the console (if specified)
    if model.log:
        model.logger.addHandler(logging.StreamHandler())
        
    model.logger.info('')

    model.logger.info('██╗   ██╗██████╗ ██████╗ ████████╗███████╗███╗   ███╗██████╗  ██████╗        ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗')
    model.logger.info('██║   ██║██╔══██╗██╔══██╗╚══██╔══╝██╔════╝████╗ ████║██╔══██╗██╔═══██╗      ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝')
    model.logger.info('██║   ██║██████╔╝██████╔╝   ██║   █████╗  ██╔████╔██║██████╔╝██║   ██║█████╗██║   ██║██║   ██║███████║██╔██╗ ██║   ██║')   
    model.logger.info('╚██╗ ██╔╝██╔═══╝ ██╔══██╗   ██║   ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██║   ██║╚════╝██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║')   
    model.logger.info(' ╚████╔╝ ██║     ██║  ██║   ██║   ███████╗██║ ╚═╝ ██║██║     ╚██████╔╝      ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║')   
    model.logger.info('  ╚═══╝  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝      ╚═════╝        ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝')                                                                                                                     
    model.logger.info('-----------------------------------------------------------------------')
    model.logger.info('Temporally Encoded Spiking Neural Network for Visual Place Recognition v1.1.0')
    model.logger.info('Queensland University of Technology, Centre for Robotics')
    model.logger.info('')
    model.logger.info('© 2023 Adam D Hines, Peter G Stratton, Michael Milford, Tobias Fischer')
    model.logger.info('MIT license - https://github.com/QVPR/VPRTempo')
    model.logger.info('\\\\\\\\\\\\\\\\\\\\\\\\')
    model.logger.info('')
    model.logger.info('Quantization enabled')
    model.logger.info('Current device is: CPU')
    model.logger.info('')