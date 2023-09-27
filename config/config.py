import os
import gc
import torch
import logging
from datetime import datetime

def configure(model):
    model.dataset = 'nordland'
    model.dataset_file = './dataset/'+model.dataset+'.csv'
    model.trainingPath = '/home/adam/data/nordland/'
    model.testPath = '/home/adam/data/nordland/'
    model.number_modules = 1
    model.number_training_images = 100
    model.number_testing_images = 100
    model.locations = ["spring", "fall"]
    model.test_location = "summer"
    model.filter = 8
    model.validation = True
    model.log = False
    
    model.training_dirs = []
    for n in model.locations:
        model.training_dirs.append(os.path.join(model.trainingPath,n))

    assert (len(model.dataset) != 0), "Dataset not defined, see README.md for details on setting up images"
    assert (os.path.isdir(model.trainingPath)), "Training path not set or path does not exist, specify for model.trainingPath"
    assert (os.path.isdir(model.testPath)), "Test path not set or path does not exist, specify for model.testPath"
    assert (os.path.isdir(model.trainingPath + model.locations[0])), "Images must be organized into folders based on locations, see README.md for details"
    assert (os.path.isdir(model.testPath + model.test_location)), "Images must be organized into folders based on locations, see README.md for details"

    model.patches = 7
    model.dims = [28,28]
    model.input = int(model.dims[0]*model.dims[1])
    model.feature = int(model.input*2)
    model.output = int(model.number_training_images/model.number_modules)
    model.intensity = 255
    model.location_repeat = len(model.locations)
    model.layers =[]
    
    model.epoch = 4
    model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model.device.type == "cuda":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.cuda.set_device(model.device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.init()
        torch.cuda.synchronize(device=model.device)
    model.T = int((model.number_training_images / model.number_modules) * model.location_repeat)
    model.annl_pow = 2
    model.imgs = {'training': [], 'testing': []}
    model.ids = {'training': [], 'testing': []}
    model.spike_rates = {'training': [], 'testing': []}
    
    model.test_true = False
    
    with open('./' + model.dataset + '_imageNames.txt') as file:
        model.imageNames = [line.rstrip() for line in file]
        
    model.filteredNames = []
    for n in range(0, len(model.imageNames), model.filter):
        model.filteredNames.append(model.imageNames[n])
    del model.filteredNames[model.number_training_images:len(model.filteredNames)]
    
    model.fullTrainPaths = []
    for n in model.locations:
        model.fullTrainPaths.append(model.trainingPath + n + '/')
    
    #now = datetime.now()
    #model.output_folder = './output/' + now.strftime("%d%m%y-%H-%M-%S")
    #os.mkdir(model.output_folder)
    
    #model.logger = logging.getLogger("VPRTempo")
    #model.logger.setLevel(logging.DEBUG)
    #logging.basicConfig(filename=model.output_folder + "/logfile.log",
     #                   filemode="a+",
      #                  format="%(asctime)-15s %(levelname)-8s %(message)s")
    #if model.log:
    #    model.logger.addHandler(logging.StreamHandler())
    
    #model.logger.info('////////////')
    #model.logger.info('VPRTempo - Temporally Encoded Visual Place Recognition v1.1.0-alpha')
    #model.logger.info('Queensland University of Technology, Centre for Robotics')
    #model.logger.info('')
    #model.logger.info('© 2023 Adam D Hines, Peter G Stratton, Michael Milford, Tobias Fischer')
    #model.logger.info('MIT license - https://github.com/QVPR/VPRTempo')
    #model.logger.info('\\\\\\\\\\\\\\\\\\\\\\\\')
    #model.logger.info('')
    #model.logger.info('CUDA available: ' + str(torch.cuda.is_available()))
    #if torch.cuda.is_available():
    #    current_device = torch.cuda.current_device()
    #    model.logger.info('Current device is: ' + str(torch.cuda.get_device_name(current_device)))
    #else:
    #    model.logger.info('Current device is: CPU')
    #model.logger.info('')
    #model.logger.info("~~ Hyperparameters ~~")
    #model.logger.info('')
    #model.logger.info('Firing threshold max: ' + str(model.thr)
    #model.logger.info('Initial STDP learning rate: ' + str(model.n_init))
    #model.logger.info('Intrinsic threshold plasticity learning rate: ' + str(model.n_itp))
    #model.logger.info('Firing rate range: [' + str(model.f_rate[0]) + ', ' + str(model.f_rate[1]) + ']')
    #model.logger.info('Excitatory connection probability: ' + str(model.p_exc))
    #model.logger.info('Inhibitory connection probability: ' + str(model.p_inh))
    #model.logger.info('Constant input: ' + str(model.c))
    #model.logger.info('')
    #model.logger.info("~~ Training and testing conditions ~~")
    #model.logger.info('')
    #model.logger.info('Number of training images: ' + str(model.number_training_images))
    #model.logger.info('Number of testing images: ' + str(model.number_testing_images))
    #model.logger.info('Number of training epochs: ' + str(model.epoch))
    #model.logger.info('Number of modules: ' + str(model.number_modules))
    #model.logger.info('Dataset used: ' + str(model.dataset))
    #model.logger.info('Training locations: ' + str(model.locations))
    #model.logger.info('Testing location: ' + str(model.test_location))
    #model.training_out = './weights/' + str(model.input_layer) + 'i' + str(model.feature_layer) + 'f' + str(model.output_layer) + 'o' + str(model.epoch) + '/'
