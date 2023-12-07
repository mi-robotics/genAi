import torch 
import numpy as np
from networks.unet import UNet
from utils.distributions import normal_kl, discretized_gaussian_loglik
from utils.functions import flat_mean
import math

from .ddpm import DDPM, BetaScheduler


class VPSDE():
    def __init__(self, config, beta_sched):
        self.config = config
        self.beta_sched:BetaScheduler = beta_sched
        self.beta_0 = self.beta_sched.start * self.beta_sched.timesteps
        self.beta_1 = self.beta_sched.end * self.beta_sched.timesteps

        self.T = 1 #TODO

    def forward(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t
        scalar = torch.sqrt(beta_t)
        return drift, scalar
    
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std
    
    def prior_sample(self, shape, generator=None):
        return torch.randm(*shape, generator=generator)
    
    def prior_log_prob(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
        z: latent code
        Returns:
        log probability density
        """
        raise "Not Implimented"
    
    def discritize(self, x, t):
        """
        DDPM discretization.
        - Uses the standard forward process paramterization
        - representing this parameterization in term of the forward SDE, eg. dx
        - Note the drift is independant of the noise
        """
        betas = self.beta_sched.get_betas(t)
        scale = sqrt_beta = torch.sqrt(betas)
        sqrt_alpha = torch.sqrt(self.beta_sched.get_alpha_bars(t))

        drift = sqrt_alpha[:, None, None, None] * x - x #change in x/dt independant of noise

        return drift, scale
    
    def reverse(self, score_fn, probability_flow):
        '''
        Provides the reverse SDE/ODE
        '''
        rsde = RSDE(self, score_fn, probability_flow, self.beta_sched.timesteps)
      
class RSDE():
    def __init__(self, sde, score_fn, probability_flow, timesteps):
        self.sde:VPSDE = sde
        self.score_fn = score_fn
        self.probability_flow = probability_flow
        self.timesteps = timesteps
        self.T = 1

    def forward(self, x, t):
        """
        reverse SDE: [f(x,t) - g(t)^2 * score]dt + g(t)*noise
        reverse ODE: [f(x,t) - g(t)^2 * score * 0.5]dt
        """
        drift, scale = self.sde.forward(x, t)
        score = self.score_fn(x,t)

        reverse_drift = drift - scale[:, None, None, None]**2 * score 
        reverse_drift = reverse_drift * 0.5 if self.probability_flow else 1.0
        reverse_scale = 0.0 if self.probability_flow else scale

        return reverse_drift, reverse_scale
    
    def discretize(self, x, t):
        """
        DDPM discretization.
        - Uses the standard reverse process paramterization
        - representing this parameterization in terms of the reverse SDE, eg. dx
        reverse SDE: [f(x,t) - g(t)^2 * score]dt + g(t)*noise
        reverse ODE: [f(x,t) - g(t)^2 * score * 0.5]dt
        """
        drift, scale = self.sde.discritize(x, t)
        score = self.score_fn(x,t)

        reverse_drift = drift - scale[:, None, None, None]**2 * score 
        reverse_drift = reverse_drift * 0.5 if self.probability_flow else 1.0
        reverse_scale = 0.0 if self.probability_flow else scale

        return reverse_drift, reverse_scale



class SGM(DDPM):

    def __init__(self, config):
        super().__init__(config)
        self.use_reduce_mean = config['use_reduce_mean'] #If losses should take mean over data dims else SUM
        
        if self.use_reduce_mean:
            self.reducer = torch.mean
        else:
            self.reducer = torch.sum

        #set up SDE
        self.sde = VPSDE(config, self.beta_sched)

    def score_function(self, x, t):
        """
        We can actually consideer the model as a noise predicitons models
        but when normalized by the vairance it becomes the score function
        """
        labels = t * 999 #converts the time to near int index of time for embedding 
        score = self.net(x, labels)
        std = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
        #TODO i beleive this devision counter acts the application variance in the applied noise  
        return -score/std[:, None,None,None] #TODO this does not generalize to different data sizes

    def forward(self, x):
        input_dict = self._extract_input_from_dataset(x)
        means, stds = self.sde.marginal_prob(input_dict['x_0'], input_dict['t'])
        input_dict['stds'] = stds
        input_dict['x_t'] = means + stds[:, None, None, None] * input_dict['noise'] #fix this projection
        input_dict['preds'] = self.score_function(input_dict['x_0'], input_dict['t'])
        raise input_dict

    def loss(self, res_dict, x):
        """
        Compute the score matching loss objective
        """
        losses = torch.square(res_dict['preds']*res_dict['stds'][:, None,None,None] * res_dict['noise'])
        losses = self.reducer(losses.reshape(losses.size(0), -1), dim=-1)
        
        loss_dict = {'loss': torch.mean(losses)}
        raise loss_dict
