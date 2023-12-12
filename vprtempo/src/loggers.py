import os
import torch
import logging

from datetime import datetime

def model_logger(): 
    """
    Configure the model logger
    """   
    now = datetime.now()
    output_folder = './vprtempo/output/' + now.strftime("%d%m%y-%H-%M-%S")
    
    os.mkdir(output_folder)
    # Create the logger
    logger = logging.getLogger("VPRTempo")
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # Set the logger level
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename=output_folder + "/logfile.log",
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    # Add the logger to the console (if specified)
    logger.addHandler(logging.StreamHandler())
        
    logger.info('')
    logger.info('██╗   ██╗██████╗ ██████╗ ████████╗███████╗███╗   ███╗██████╗  ██████╗') 
    logger.info('██║   ██║██╔══██╗██╔══██╗╚══██╔══╝██╔════╝████╗ ████║██╔══██╗██╔═══██╗')
    logger.info('██║   ██║██████╔╝██████╔╝   ██║   █████╗  ██╔████╔██║██████╔╝██║   ██║')
    logger.info('╚██╗ ██╔╝██╔═══╝ ██╔══██╗   ██║   ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██║   ██║')
    logger.info(' ╚████╔╝ ██║     ██║  ██║   ██║   ███████╗██║ ╚═╝ ██║██║     ╚██████╔╝')
    logger.info('  ╚═══╝  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝      ╚═════╝ ')
    logger.info('-----------------------------------------------------------------------')
    logger.info('Temporally Encoded Spiking Neural Network for Visual Place Recognition v1.1.0')
    logger.info('Queensland University of Technology, Centre for Robotics')
    logger.info('')
    logger.info('© 2023 Adam D Hines, Peter G Stratton, Michael Milford, Tobias Fischer')
    logger.info('MIT license - https://github.com/QVPR/VPRTempo')
    logger.info('\\\\\\\\\\\\\\\\\\\\\\\\')
    logger.info('')
    if torch.cuda.is_available():
        logger.info('CUDA available: ' + str(torch.cuda.is_available()))
        current_device = torch.cuda.current_device()
        logger.info('Current device is: ' + str(torch.cuda.get_device_name(current_device)))
    else:
        logger.info('CUDA available: ' + str(torch.cuda.is_available()))
        logger.info('Current device is: CPU')
    logger.info('')

def model_logger_quant(): 
    """
    Configure the logger
    """   

    now = datetime.now()
    output_folder = './vprtempo/output/' + now.strftime("%d%m%y-%H-%M-%S")
    
    os.mkdir(output_folder)
    # Create the logger
    logger = logging.getLogger("VPRTempo")
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # Set the logger level
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename=output_folder + "/logfile.log",
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    # Add the logger to the console (if specified)
    logger.addHandler(logging.StreamHandler())
        
    logger.info('')

    logger.info('██╗   ██╗██████╗ ██████╗ ████████╗███████╗███╗   ███╗██████╗  ██████╗        ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗')
    logger.info('██║   ██║██╔══██╗██╔══██╗╚══██╔══╝██╔════╝████╗ ████║██╔══██╗██╔═══██╗      ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝')
    logger.info('██║   ██║██████╔╝██████╔╝   ██║   █████╗  ██╔████╔██║██████╔╝██║   ██║█████╗██║   ██║██║   ██║███████║██╔██╗ ██║   ██║')   
    logger.info('╚██╗ ██╔╝██╔═══╝ ██╔══██╗   ██║   ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██║   ██║╚════╝██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║')   
    logger.info(' ╚████╔╝ ██║     ██║  ██║   ██║   ███████╗██║ ╚═╝ ██║██║     ╚██████╔╝      ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║')   
    logger.info('  ╚═══╝  ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝      ╚═════╝        ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝')                                                                                                                     
    logger.info('-----------------------------------------------------------------------')
    logger.info('Temporally Encoded Spiking Neural Network for Visual Place Recognition v1.1.0')
    logger.info('Queensland University of Technology, Centre for Robotics')
    logger.info('')
    logger.info('© 2023 Adam D Hines, Peter G Stratton, Michael Milford, Tobias Fischer')
    logger.info('MIT license - https://github.com/QVPR/VPRTempo')
    logger.info('\\\\\\\\\\\\\\\\\\\\\\\\')
    logger.info('')
    logger.info('Quantization enabled')
    logger.info('Current device is: CPU')
    logger.info('')
