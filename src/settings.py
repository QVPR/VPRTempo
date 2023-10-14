import os
import torch
import logging
import csv

from datetime import datetime

def configure(model):
    """
    Configure the model
    """
    model.dataset = 'nordland' # Dataset name
    model.dataset_file = './dataset/'+model.dataset+'.csv' # Dataset file (must be PyTorch Dataset  )
    model.trainingPath = './dataset/' # Path to training images
    model.testPath = './dataset/' # Path to testing images
    model.number_modules = 1 # Number of expert modules (currently not implemented)
    model.number_training_images = 500 # Number of training images
    model.number_testing_images = 500 # Number of testing images
    model.locations = ["spring","fall"] # Locations to train on (location repeats for training datasets)
    model.test_locations = ["summer"] # Location to query with
    model.filter = 8 # Filter for training images
    model.validation = True # Validation (maybe deprecated for now?)
    model.log = True # Log to console
    
    # Output the training and testing directories
    model.training_dirs = []
    for n in model.locations:
        model.training_dirs.append(os.path.join(model.trainingPath,n))    
    model.testing_dirs = []
    for n in model.test_locations:
        model.testing_dirs.append(os.path.join(model.testPath,n))
    
    # Check that the dataset is defined properly
    assert (len(model.dataset) != 0), "Dataset not defined, see README.md for details on setting up images"
    assert (os.path.isdir(model.trainingPath)), "Training path not set or path does not exist, specify for model.trainingPath"
    assert (os.path.isdir(model.testPath)), "Test path not set or path does not exist, specify for model.testPath"
    assert (os.path.isdir(model.trainingPath + model.locations[0])), "Images must be organized into folders based on locations, see README.md for details"
    assert (os.path.isdir(model.testPath + model.test_locations[0])), "Images must be organized into folders based on locations, see README.md for details"
    
    # Set the model parameters
    model.epoch = 4 # Number of epochs
    model.patches = 7 # Number of patches
    model.dims = [28,28] # Dimensions of the input image
    model.location_repeat = len(model.locations) # Number of times to repeat the locations
    model.annl_pow = 2 # Power of the annealmeant function

    """
    These parameters are used to define the network architecture
    """
    model.input = int(model.dims[0]*model.dims[1]) # Number of input neurons
    model.feature = int(model.input*2) # Number of feature neurons
    model.output = int(model.number_training_images/model.number_modules) # Number of output neurons
    
    # Set the torch device
    #model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.device = torch.device("cpu")
    if model.device.type == "cuda":
        torch.cuda.init()
        torch.cuda.synchronize(device=model.device)

    # Determine the total number of timesteps across training images, modules, and location repeats
    model.T = int((model.number_training_images / model.number_modules) * model.location_repeat)

def image_csv(model):
    """
    Load the image names from the CSV file and filter them
    """
    # Load the image names from the CSV file
    with open(os.path.join('./dataset', model.dataset + '.csv'), mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        model.imageNames = [row[0] for row in reader]
    # Remove the header
    del model.imageNames[0]
    # Filter the image names
    model.filteredNames = []
    for n in range(0, len(model.imageNames), model.filter):
        model.filteredNames.append(model.imageNames[n])
    # Remove the training images from the filtered names
    del model.filteredNames[model.number_training_images:len(model.filteredNames)]
    # Store the full training paths
    model.fullTrainPaths = []
    for n in model.locations:
        model.fullTrainPaths.append(model.trainingPath + n + '/')

def model_logger(model): 
    """
    Configure the model logger
    """   
    # Create the output folder
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
    model.logger.info('CUDA available: ' + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        model.logger.info('Current device is: ' + str(torch.cuda.get_device_name(current_device)))
    else:
        model.logger.info('Current device is: CPU')
    model.logger.info('')