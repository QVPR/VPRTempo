import os
import torch
import logging

from datetime import datetime

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
    if torch.cuda.is_available():
        model.logger.info('CUDA available: ' + str(torch.cuda.is_available()))
        device = "cuda:0"
        current_device = torch.cuda.current_device()
        model.logger.info('Current device is: ' + str(torch.cuda.get_device_name(current_device)))
    else:
        model.logger.info('CUDA available: ' + str(torch.cuda.is_available()))
        model.logger.info('Current device is: CPU')
        device = "cpu"
    model.logger.info('')

    return device

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
    model.logger.addHandler(logging.StreamHandler())

    device = "cpu"
        
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

    return device