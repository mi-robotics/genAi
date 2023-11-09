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

from vae.vae import VAE

from tqdm import tqdm

class ModelTrainer():

    def __init__(self, config, model):

        self.model:torch.nn.Module = model
        self.dataset = config['dataset']
        self.epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.lr_scheduler = config['lr_scheduler']
        self.optimizer = config['optimzer']
        self.load_dataset()
        self.init_optim()

        return
    
    def load_dataset(self):
        if self.dataset == 'digits':
            self.dataset = Digits(mode='train')

    def init_optim(self):
        if self.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    
    def train_model(self):

        #iterate over epochs
        pbar = tqdm(range(self.epochs), desc="Epoch", position=0)
        for epoch in pbar:

            epoch_store = {
                'loss':[],
                'elbo':[],
                'dkl':[]
            }

            #iterate over batches
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

            for mb in tqdm(data_loader, desc="Batch", position=1, leave=False):
                
                res_dict = self.model(mb)
                loss_dict:torch.Tensor = self.model.loss(res_dict, mb)

                self.optimizer.zero_grad()
                loss_dict['loss'].backward()
                self.optimizer.step()

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

    config = {

        "dataset":"digits",
        "num_epochs":500,
        "batch_size":100,
        "learning_rate":1e-3,
        "lr_scheduler":None,
        "optimzer":"adam",

        "data_distribution":'categorical',
        "data_dims":64,
        'num_classes':17
    }

    model = VAE(config)

    trainer = ModelTrainer(config, model=model)
    trainer.train_model()

