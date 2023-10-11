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
import torch
import gc
import sys
sys.path.append('./src')
sys.path.append('./models')
sys.path.append('./settings')
sys.path.append('./output')
sys.path.append('./dataset')
sys.path.append('./config')
torch.multiprocessing.set_sharing_strategy("file_system")
import blitnet as bn
import utils as ut
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization

from config import configure, image_csv, model_logger
from dataset import CustomImageDataset, SetImageAsSpikes, ProcessImage
from torch.utils.data import DataLoader
from torch.ao.quantization import QuantStub, DeQuantStub
from tqdm import tqdm

class VPRTempo(nn.Module):
    def __init__(self):
        super(VPRTempo, self).__init__()

        # Configure the network
        configure(self)
        
        # Define the images to load (both training and inference)
        image_csv(self)

        # Common model architecture
        self.feature_layer = bn.SNNLayer(
            dims=[self.input, self.feature],
            thr_range=[0, 0.5],
            fire_rate=[0.2, 0.9],
            ip_rate=0.15,
            stdp_rate=0.005,
            const_inp=[0, 0.1],
            p=[0.1, 0.5]
        )
        self.output_layer = bn.SNNLayer(
            dims=[self.feature, self.output],
            ip_rate=0.15,
            stdp_rate=0.005,
            spk_force=True
        )
        self.layers = {
            'layer0': self.feature_layer,
            'layer1': self.output_layer
        }
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        self.add = nn.quantized.FloatFunctional()
        
    def model_logger(self):
        # Start the model logger
        model_logger(self)

    def _anneal_learning_rate(self, layer, mod, itp, stdp):
        if np.mod(mod, 10) == 0:
            pt = pow(float(self.T - mod) / self.T, self.annl_pow)
            layer.eta_ip = torch.mul(itp, pt)
            layer.eta_stdp = torch.mul(stdp, pt)
            
        return layer

    def train_model(self,layer,train_loader,logger,prev_layer=None):
    
        # Initialize the tqdm progress bar
        pbar = tqdm(total=int(self.T * self.epoch),
                    desc="Training ",
                    position=0)
        
        init_itp = layer.eta_ip.detach()
        init_stdp = layer.eta_stdp.detach()
        
        for epoch in range(self.epoch):
            mod = 0  # Used to determine the learning rate annealment, resets at each epoch

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                idx = labels / self.filter
                if not prev_layer == None:
                    # Get the spikes to train the next layer
                    prev_layer.x_input = self.forward(images,prev_layer)
                    out = bn.clamp_spikes(prev_layer)
                    images = out
                # Run forward pass
                layer = bn.add_input(layer)
                layer.x_input = self.forward(images, layer)
                out = bn.clamp_spikes(layer)
                layer = bn.calc_stdp(images, out, layer, idx, prev_layer=prev_layer)
                
                # Adjust learning rates
                layer = self._anneal_learning_rate(layer, mod, init_itp, init_stdp)

                # Reset x and x_input for next iteration
                layer.x.fill_(0.0)
                layer.x_input.fill_(0.0)
                mod += 1
                pbar.update(1)

        # Close the tqdm progress bar
        pbar.close()
    
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    
    def evaluate(self, test_loader, layers=None):
        numcorr = 0
        idx = 0
            
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.number_testing_images,
                    desc="Running the test network",
                    position=0)
    
        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            for layer in layers:
                images = self.forward(images, layer)
    
            if torch.argmax(images.reshape(1, self.number_training_images)) == idx:
                numcorr += 1
    
            idx += 1
            
            pbar.update(1)
    
        pbar.close()
        accuracy = round((numcorr/self.number_testing_images)*100,2)
        model.logger.info("P@100R: "+ str(accuracy) + '%')

        return accuracy

    
    def forward(self, spikes, layer):
        """
        Compute the forward pass of the model.
    
        Parameters:
        - spikes (Tensor): Input spikes.
    
        Returns:
        - Tensor: Output after processing.
        """
        spikes = self.quant(spikes)
        layer.x_input = self.quant(layer.x_input)
        layer.x_input = bn.calc_spikes(spikes, layer, self.add)
        layer.x_input = self.dequant(layer.x_input)
        spikes = self.dequant(spikes)

        return layer.x_input
    
    def save_model(self, model_out):    
        """Save the trained model to models output folder."""
        torch.save(self.state_dict(), model_out) 
        
    def load_model(self, model_path):
        """Load pre-trained model and set the state dictionary keys."""
        self.load_state_dict(torch.load(model_path, map_location=self.device),
                             strict=False)
        self.eval()
            
def generate_model_name(model):
    """Generate the model name based on its parameters."""
    return ("VPRTempo" +
            str(model.input) +
            str(model.feature) +
            str(model.output) +
            str(model.number_modules) +
            '.pth')

def check_pretrained_model(model_name):
    """Check if a pre-trained model exists and prompt the user to retrain if desired."""
    if os.path.exists(os.path.join('./models', model_name)):
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        retrain = input(prompt).strip().lower()
        return retrain == 'n'
    return False

def train_new_model(model, model_name, qconfig):
    set_image_as_spikes = SetImageAsSpikes(test=False)
    set_image_as_spikes.eval()
    set_image_as_spikes.train()
    image_transform = ProcessImage(model.dims, model.patches)
    train_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                       img_dirs=model.training_dirs,
                                       transform=image_transform,
                                       skip=model.filter,
                                       max_samples=model.number_training_images,
                                       test=False)
    train_loader = DataLoader(train_dataset, 
                              batch_size=1, 
                              shuffle=False,
                              num_workers=1,
                              persistent_workers=True)
    trainer = VPRTempo()
    trainer.train()
    trainer.to('cpu')
    trainer.feature_layer.qconfig = qconfig
    trainer.output_layer.qconfig = qconfig
    trainer = quantization.prepare_qat(trainer, inplace=True)
    trainer.train_model(trainer.feature_layer, train_loader, model.logger)
    trainer.train_model(trainer.output_layer, train_loader, model.logger,
                        prev_layer=trainer.feature_layer)
    trainer.eval()
    trainer.save_model(os.path.join('./models', model_name))

def run_inference(model, model_name, qconfig):
    set_image_as_spikes = SetImageAsSpikes(test=True)
    set_image_as_spikes.eval() 
    image_transform = ProcessImage(model.dims, model.patches)
    test_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                      img_dirs=model.testing_dirs,
                                      transform=image_transform,
                                      skip=model.filter,
                                      max_samples=model.number_testing_images)
    test_loader = DataLoader(test_dataset, 
                             batch_size=1, 
                             shuffle=False,
                             num_workers=1,
                             persistent_workers=True)

    model = VPRTempo()
    model.to('cpu')
    model.eval()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model.feature_layer.qconfig = qconfig
    model.output_layer.qconfig = qconfig
    model = quantization.prepare(model, inplace=True)
    model.load_model(os.path.join('./models', model_name))
    model = quantization.convert(model, inplace=True)

    # Use evaluate method for inference accuracy
    model.evaluate(test_loader,
                   layers=[model.feature_layer, model.output_layer]
                   )

if __name__ == "__main__":
    # Temporary place holder for now, weird multiprocessing bug with larger models
    torch.set_num_threads(1)

    # Initialize the model and image transforms
    model = VPRTempo()
    model.model_logger()
    qconfig = quantization.get_default_qat_qconfig('fbgemm')
    model_name = generate_model_name(model)
    use_pretrained = check_pretrained_model(model_name)
    
    if not use_pretrained:
        train_new_model(model, model_name, qconfig)
    with torch.no_grad():    
        run_inference(model, model_name, qconfig)   