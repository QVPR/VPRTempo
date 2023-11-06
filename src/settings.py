import os
import torch
import logging

from datetime import datetime

def configure(model):
    """
    Configure the model
    """
    model.dataset = 'nordland' # Dataset name
    model.dataset_file = os.path.join('./dataset',model.dataset+'.csv') # Dataset file (must be PyTorch Dataset)
    model.trainingPath = './dataset/' # Path to training images
    model.testPath = './dataset/' # Path to testing images
    model.number_modules = 1 # Number of expert modules (currently not implemented)
    model.number_training_images = 250 # Number of training images
    model.number_testing_images = model.number_training_images # Number of testing images
    model.locations = ["spring"] # Locations to train on (location repeats for training datasets)
    model.test_locations = ["winter"] # Location to query with
    model.filter = 8 # Filter for training images
    model.validation = True # Validation (maybe deprecated for now?)
    model.log = True # Log to console
    model.quantize = False # Quantize the network
    
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

    # Set the model parameters
    model.epoch = 8 # Number of epochs
    model.patches = 15 # Number of patches
    model.dims = [56,56] # Dimensions of the input image
    model.location_repeat = len(model.locations) # Number of times to repeat the locations
    model.annl_pow = 2 # Power of the annealmeant function

    """
    These parameters are used to define the network architecture
    """
    model.input = int(model.dims[0]*model.dims[1]) # Number of input neurons
    model.feature = int(model.input*2) # Number of feature neurons
    model.output = int(model.number_training_images/model.number_modules) # Number of output neurons
    
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