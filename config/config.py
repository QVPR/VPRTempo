import os
import torch
import logging
import csv

from datetime import datetime

def configure(model):
    model.dataset = 'nordland'
    model.dataset_file = './dataset/'+model.dataset+'.csv'
    model.trainingPath = '/home/adam/data/nordland/'
    model.testPath = '/home/adam/data/nordland/'
    model.number_modules = 1
    model.number_training_images = 250
    model.number_testing_images = 250
    model.locations = ["spring","fall"]
    model.test_locations = ["summer"]
    model.filter = 8
    model.validation = True
    model.log = True
    
    model.training_dirs = []
    for n in model.locations:
        model.training_dirs.append(os.path.join(model.trainingPath,n))
        
    model.testing_dirs = []
    for n in model.test_locations:
        model.testing_dirs.append(os.path.join(model.testPath,n))
        
    assert (len(model.dataset) != 0), "Dataset not defined, see README.md for details on setting up images"
    assert (os.path.isdir(model.trainingPath)), "Training path not set or path does not exist, specify for model.trainingPath"
    assert (os.path.isdir(model.testPath)), "Test path not set or path does not exist, specify for model.testPath"
    assert (os.path.isdir(model.trainingPath + model.locations[0])), "Images must be organized into folders based on locations, see README.md for details"
    assert (os.path.isdir(model.testPath + model.test_locations[0])), "Images must be organized into folders based on locations, see README.md for details"
    
    model.epoch = 4
    model.patches = 7
    model.dims = [28,28]
    model.input = int(model.dims[0]*model.dims[1])
    model.feature = int(model.input*2)
    model.output = int(model.number_training_images/model.number_modules)
    model.intensity = 1
    model.location_repeat = len(model.locations)
    model.layers = []
    
    #model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.device = torch.device("cpu")
    if model.device.type == "cuda":
        torch.cuda.init()
        torch.cuda.synchronize(device=model.device)
    model.T = int((model.number_training_images / model.number_modules) * model.location_repeat)
    model.annl_pow = 2

def image_csv(model):
    with open(os.path.join('./dataset', model.dataset + '.csv'), mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        model.imageNames = [row[0] for row in reader]
    del model.imageNames[0]
        
    model.filteredNames = []
    for n in range(0, len(model.imageNames), model.filter):
        model.filteredNames.append(model.imageNames[n])
    del model.filteredNames[model.number_training_images:len(model.filteredNames)]

def model_logger(model):    
    model.fullTrainPaths = []
    for n in model.locations:
        model.fullTrainPaths.append(model.trainingPath + n + '/')
    
    now = datetime.now()
    model.output_folder = './output/' + now.strftime("%d%m%y-%H-%M-%S")
    os.mkdir(model.output_folder)
    
    model.logger = logging.getLogger("VPRTempo")
    if (model.logger.hasHandlers()):
        model.logger.handlers.clear()
    model.logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename=model.output_folder + "/logfile.log",
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
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