import os
import sys

# Get the current directory (where this script is located)
current_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (root of your code base)
code_base_root = os.path.dirname(current_dir)

# Add the code base root to sys.path
sys.path.append(code_base_root)


import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.digits import Digits
from datasets.cifar10 import CIFAR10
from utils.ema import EMA
from utils.modelCheckpoint import Checkpoint

from vae.vae import VAE
from diffussion.ddpm import DDPM

from tqdm import tqdm

class ModelTrainer():

    def __init__(self, config, model):

        self.model:torch.nn.Module = model
        self.ema = EMA(self.model) if config['use_ema'] else None
        self.lr_scheduler = None #TODO: not implimented

        self.dataset = config['dataset']
        self.epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.lr_scheduler = config['lr_scheduler']
        self.optimizer = config['optimzer']

        self._init_dataset()
        self._init_optim()
        self._init_scheduler()

        self.checkpoint = Checkpoint(config, self._get_stated_moduels())

        return
    
    def _init_dataset(self):
        if self.dataset == 'digits':
            self.dataset = Digits(mode='train')
        if self.dataset == 'cifar10':
            self.dataset = CIFAR10(root=os.path.expanduser("~/datasets"), mode='train')

    def _init_scheduler(self):
        self.lr_scheduler = None

    def _init_optim(self):
        if self.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def _get_stated_moduels(self):
        mods = [self.model, self.lr_scheduler, self.optimizer, self.ema]
        return [x for x in mods if x is not None]

    
    def train_model(self):

        #iterate over epochs
        pbar = tqdm(range(self.epochs), desc="Epoch", position=0)
        for epoch in pbar:

            #TODO: Use a stats tracker
            # - Must be refactored to account for different models
            epoch_store = {
                'loss':[],
                'elbo':[],
                'dkl':[]
            }

            #iterate over batches
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

            for mb in tqdm(data_loader, desc="Batch", position=1, leave=False):
                print('batch size', mb.size())
          
                res_dict = self.model(mb)
                loss_dict:torch.Tensor = self.model.loss(res_dict, mb)

                self.optimizer.zero_grad()
                loss_dict['loss'].backward()
                self.optimizer.step()

                print("got here safely")
                quit()

                epoch_store['loss'].append(loss_dict['loss'].detach())
                epoch_store['elbo'].append(loss_dict['elbo'].detach())
                epoch_store['dkl'].append(loss_dict['dkl'].detach())

                # Update the progress bar with the current epoch and loss value
                pbar.set_postfix({
                    "Mean Loss": np.mean(epoch_store['loss']), 
                    "Mean ELBO": np.mean(epoch_store['elbo']),
                    "Mean D_kl": np.mean(epoch_store['dkl'])})

            print('loss', np.mean(epoch_store['loss']))
            self.model.generate_samples()
                

            
            

        return 
    


if __name__ == '__main__':

    # config = {
    #     "model":"vae",

    #     "dataset":"digits",
        
    #     "num_epochs":500,
    #     "batch_size":100,
    #     "learning_rate":1e-3,
    #     "lr_scheduler":None,
    #     "optimzer":"adam",

    #     "data_distribution":'categorical',
    #     "data_dims":64,
    #     'num_classes':17
    # }

    # model = VAE(config)

    config = {
        "model":"ddpm",
        "dataset":"cifar10",

        #Tracking
        "use_wandb":False,
        "project_name":"ddpm-experiments",
        "experiment_name":None,
        "use_ema":True,
        "ema_decay":None,
        "track_values":{
            "loss":None
        },

        #ddpm
        "timesteps":1000,
        "beta_end": 0.02,
        "beta_start": 0.0001,
        "beta_scheduler":'linear',

        "model_var_type":"fixed-small",
        "vdm_type":"noise",
        "loss_type":"mse",

        "num_epochs":50,
        "batch_size":128,
        "learning_rate":1e-3,
        "lr_scheduler":None,
        "optimzer":"adam",

        #unet params
        "network":'unet',
        'unet':{
            "in_channels": 3,
            "hid_channels": 128,
            "out_channels": 3,
            "ch_multipliers": [1, 2, 2, 2],
            "num_res_blocks": 2,
            "apply_attn": [False, True, False, False],
            "drop_rate": 0.1
        }
    }

    model = DDPM(config)
    trainer = ModelTrainer(config, model=model)
    trainer.train_model()

