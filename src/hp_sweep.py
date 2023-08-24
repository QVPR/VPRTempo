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

import wandb
import pprint

import VPRTempo as vt
import numpy as np

# log into weights & biases
wandb.login()

# define the method and parameters for grid search
sweep_config = {'method':'grid'}
metric = {'p100r':'accuracy', 'goal':'maximize'}

sweep_config['metric'] = metric

parameters_dict = {'theta_max' : 
                       {'values' : np.ndarray.tolist(np.round(np.linspace(0.1,1,10),2))},
                   'n_init' :
                       {'values' : np.ndarray.tolist(np.round(np.linspace(0.025,0.25,10),4))},    
                   'n_itp' :
                       {'values' : np.ndarray.tolist(np.round(np.linspace(0.025,0.25,10),4))},    
                   'f_rateL' :
                       {'values' : np.ndarray.tolist(np.round(np.linspace(0.01,0.1,10),4))},
                   'f_rateH' :
                       {'values' : np.ndarray.tolist(np.round(np.linspace(0.1,1,10),2))},
                   'p_exc' :
                       {'values' : np.ndarray.tolist(np.round(np.linspace(0.01,0.25,10),2))},
                   'p_inh' :
                       {'values' : np.ndarray.tolist(np.round(np.linspace(0.1,0.75,10),2))},
                    'c' :
                       {'values' : np.ndarray.tolist(np.round(np.linspace(0.1,0.5,10),2))} 
                       }

sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)

# start sweep controller
sweep_id = wandb.sweep(sweep_config, project="vprtempo-grid-search")

# load the testing and training data for VPRTempo
model = vt.snn_model()

# initialize w&b
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        