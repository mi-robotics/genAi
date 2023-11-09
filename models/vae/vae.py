import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

EPS = 1.e-5

class VaeEncoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dims_in = config.get('data_dims')
        self.dims_out = config.get('latent_dims', 10) * 2
        self.latent_dims = config.get('latent_dims', 10) 
        self.units = config.get('units', [128,128])

        #create basic MLP
        net = []
        units_in = self.dims_in
        for units in self.units:
            print(units_in, units)
            net.append(torch.nn.Linear(units_in, units))
            net.append(torch.nn.Tanh())
            units_in = units
        net.append(torch.nn.Linear(units_in, self.dims_out))
        self.net = torch.nn.Sequential(*net)

        #intialize network
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Initialize weights using Xavier initialization
                nn.init.zeros_(layer.bias)            

        return
       
    def forward(self, x):
        # x -> [batch_size, num_pixels]

        # [batch_sise, latent_dim * 2]
        dist = self.net(x)

        #reparamaterize
        mu = dist[:, :self.latent_dims]
        log_var = dist[:, self.latent_dims:]

        #random normal samples
        e = torch.randn(size=log_var.size())

        #get the latents from reparameterization trick 
        std = torch.exp(0.5*log_var)
        z = mu + std * e

        return {'z':z, 'mu':mu, 'log_var':log_var}
    
    def log_prob(self, z, mu, log_var ):
        """
        z, mu, log_var-> [batch_size, latent_dims]
        """
        var = torch.exp(log_var)
        log_determinant = torch.log(var).sum(1) #[batch_size]

        #size -> [batch_size]
        log_probs = - 0.5 * (self.latent_dims * torch.log(torch.Tensor([2*torch.pi])) + log_determinant + torch.sum((z - mu)/var, dim=1))

        return log_probs.mean()
    

class VaeDecoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.data_distribution = config.get('data_distribution')
        self.num_classes = config.get('num_classes', 0)
        assert (self.data_distribution == 'categorical') == (self.num_classes >= 1) 


        self.dims_in = config.get('latent_dims', 10) 
        self.dims_out = config.get('data_dims') 
        if self.num_classes:
            self.dims_out *= self.num_classes
        
        self.units = config.get('units', [128,128])

        #create basic MLP
        net = []
        units_in = self.dims_in
        for units in self.units:
            net.append(torch.nn.Linear(units_in, units))
            net.append(torch.nn.Tanh())
            units_in = units
        net.append(torch.nn.Linear(units_in, self.dims_out))
        self.net = torch.nn.Sequential(*net)

        #intialize network
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Initialize weights using Xavier initialization
                nn.init.zeros_(layer.bias)    


        return
    
    def forward(self, z):
        
        #[batch size, image_dims * num_classes]
        logits = self.net(z)

        if self.data_distribution == 'categorical':
            #[batch size, image_dims, num_classes]
            logits = torch.reshape(logits, (logits.size(0), -1, self.num_classes))
            probs = torch.softmax(logits, dim=-1)

        #depending on the distribution
        return {'probs':probs}
    
    

class GaussianPrior():

    def __init__(self, config):
        self.config = config
        self.latent_dim = config.get('latent_dims', 10)
        return 
    
    def sample(self, batch_size):
        return torch.randn(size=(batch_size, self.latent_dim))
    

    def log_prob(self, z):
        """ Used for calculating the D_KL loss """
        #z -> [batch_size, latent_dims]

        # size -> [batch_size]
        log_probs = (-self.latent_dim/2) * torch.log(torch.Tensor([2*torch.pi])) - torch.sum(z**2, dim=1)

        return log_probs.mean()
    


class VAE(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc = VaeEncoder(config)
        self.dec = VaeDecoder(config)
        self.prior = GaussianPrior(config)

        self.data_distribution = config.get('data_distribution', 'continuous')
        assert self.data_distribution in ['continuous', 'bournoli', 'categorical'] 

        self.latent_dim = config.get('latent_dims', 10)
        self.beta = config.get('beta', 0)
        self.num_classes = config.get('num_classes', 0)

        return 
    

    def forward(self, x):
        """
        - encoder the input 
        - reparameterization trick 
        - decode the latents
        """
        #[batch_size, latents_size*2]
        enc_dict = self.enc(x)

        #reconstruct the image
        dec_dict = self.dec(enc_dict['z'])

        enc_dict.update(dec_dict)

        return enc_dict
    
    def sample_categorical(self, probs):
        #probs = [batch_size, image_dims, num_cats]
        batch_size = probs.size(0)
        image_dims = probs.size(1)
        #torch multi nominal only operates over 2D space
        probs = probs.reshape((-1, self.num_classes)) #[batch_size*image_dim, num_cats]
        selection = torch.multinomial(probs, num_samples=1) #samples from distribution -> [batch_size*image_dim, num_samples]
        selection = selection.squeeze(dim=1)
        samples = selection.reshape((batch_size, image_dims))
        return samples


    def loss(self, vae_dict, truths):
        """
        - reconstruction loss
        - D_kl regularization 
        """
        probs = vae_dict['probs']
        mu = vae_dict['mu']
        log_var = vae_dict['log_var']
        z = vae_dict['z']

        #compute the rec onstruction loss -> maximise the log probability
        if self.data_distribution == 'categorical':
            # eg minimize neg log likelihood
            recon = self.log_likelood(probs, truths)
     

        elif self.data_distribution == 'continuous-gaussian':
            raise "Not Implimented"
        
        elif self.data_distribution == 'continous-mse':
            raise "Not Implimented"
        
        #compute D_KL 
        dkl = self.prior.log_prob(z) - self.enc.log_prob(z, mu, log_var)
       
        loss = - (recon + dkl)

        out = {'loss':loss, 'elbo':recon, "dkl":dkl}

        return out
    

    def log_likelood(self, preds, truths:torch.Tensor):

        preds = torch.transpose(preds, 2,1)
        preds = torch.torch.clamp(preds, EPS, 1. - EPS) #ebsyre raw probs are in the range [0,1]
        log_probs = torch.log(preds)
        nll = -torch.nn.functional.nll_loss(log_probs.float(), truths.long())

        return nll
    

    def generate_samples(self,):
  
        # GENERATIONS-------
        self.eval()

        with torch.no_grad():
            # [8, latent dim]
            z = self.prior.sample(16)
            dec_dict = self.dec.forward(z)
            x = self.sample_categorical(dec_dict['probs']).numpy()

            fig, ax = plt.subplots(4, 4)
            for i, ax in enumerate(ax.flatten()):
                plottable_image = np.reshape(x[i], (8, 8))
                ax.imshow(plottable_image, cmap='gray')
                ax.axis('off')

            plt.show()

        self.train()

        return
 


    def reconstruct_samples(self, samples):
        return 
    



    
    

