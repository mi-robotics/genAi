
import torch 
import numpy as np
from networks.unet import UNet
from utils.distributions import normal_kl, discretized_gaussian_loglik
from utils.functions import flat_mean
import math

class BetaScheduler():

    def __init__(self, start, end, method:str, timesteps:int):
        self.start = start
        self.end = end
        self.method = method
        self.timesteps = timesteps

        #noising parameterization
        self._init_betas()
        self.alphas = 1-self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars) #encoder mean coeficients
        self.one_minus_alpha_bars = 1. - self.alpha_bars
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars) #encoder std

        #for sampling the posterior encoder q(x_t-1 | x_t, x_0)
        self.alpha_bar_prevs = torch.cat([torch.Tensor([1.0,]), self.alpha_bars[:-1]])
        self.sqrt_alpha_bar_prevs = torch.sqrt(self.alpha_bar_prevs)
        self.one_minus_alpha_bar_prevs = 1.0 - self.alpha_bar_prevs

        self.mu_coef_x0s = (self.sqrt_alpha_bar_prevs*self.betas)/self.one_minus_alpha_bars
        self.mu_coes_xts = (self.alphas*self.one_minus_alpha_bar_prevs)/self.one_minus_alpha_bars
        self.var_coefs = (self.one_minus_alpha_bar_prevs/self.one_minus_alpha_bars)*self.betas

        #for predicting x_0 given predicted noise 
        self.sqrt_recip_alphas_bar = torch.sqrt(1. / self.alpha_bars)
        self.sqrt_recip_m1_alphas_bar = torch.sqrt(1. / self.alpha_bars - 1.)  # m1: minus 1

        return 
    
    def _init_betas(self, dtype=torch.float64):
        if self.method == 'quad':
            self.betas = torch.linspace(self.start ** 0.5, self.end ** 0.5, self.timesteps, dtype=dtype) ** 2
        elif self.method  == 'linear':
            self.betas = torch.linspace(self.start, self.end, self.timesteps, dtype=dtype)
        elif self.method  == 'warmup10':
            self.betas = self._warmup_beta(self.start, self.end, self.timesteps, 0.1, dtype=dtype)
        elif self.method  == 'warmup50':
            self.betas = self._warmup_beta(self.start, self.end, self.timesteps, 0.5, dtype=dtype)
        elif self.method  == 'const':
            self.betas = self.end * torch.ones(self.timesteps, dtype=dtype)
        elif self.method  == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            self.betas = 1. / torch.linspace(self.timesteps, 1, self.timesteps, dtype=dtype)
        else:
            raise NotImplementedError("Beta Scheduler")
        assert self.betas.shape == (self.timesteps, )

    def _warmup_beta(self, beta_start, beta_end, timesteps, warmup_frac, dtype):
        betas = self.end * torch.ones(self.timesteps, dtype=dtype)
        warmup_time = int(timesteps * warmup_frac)
        betas[:warmup_time] = torch.linspace(beta_start, beta_end, warmup_time, dtype=dtype)
        return betas
    

    def get_betas(self, t):
        """
        t -> [batch_size]
        """
        return self.betas[t]
    
    def get_alphas(self, t):
        """
        t -> [batch_size]
        """
        return self.alphas[t]
    
    def get_sqrt_alpha_bars(self, t):
        """
        t -> [batch_size]
        """
        return self.sqrt_alpha_bars[t].to(torch.float32)
    
    def get_alpha_bars(self, t):
        """
        t -> [batch_size]
        """
        return self.alpha_bars[t].to(torch.float32)
    
    def get_sqrt_one_minus_alpha_bars(self, t):
        """
        t -> [batch_size]
        """
        return self.sqrt_one_minus_alpha_bars[t].to(torch.float32)
    
    def get_one_minus_alpha_bars(self, t):
        """
        t -> [batch_size]
        """
        return self.one_minus_alpha_bars[t].to(torch.float32)
 
    def get_posterior_coeficients(self, t) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mu_coef_x0s[t].to(torch.float32), \
                self.mu_coes_xts[t].to(torch.float32), \
                self.var_coefs[t].to(torch.float32)
    
    def get_incerse_noising_coeficients(self, t) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sqrt_recip_alphas_bar[t].to(torch.float32), \
                self.sqrt_recip_m1_alphas_bar[t].to(torch.float32)



class DDPM(torch.nn.Module):
    """
    @timesteps: the number of diffussion steps
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self._init_model()
        self.model_var_type = config['model_var_type']
        self.timesteps = config['timesteps']
        self.loss_type = config['loss_type']

        self.vdm_type = config['vdm_type'] # see https://arxiv.org/pdf/2208.11970.pdf
        self.beta_sched = BetaScheduler(config['beta_start'], config['beta_end'], config['beta_scheduler'], self.timesteps)
        
        # maintain a process-specific generator -> on cpu
        self.generator = torch.Generator().manual_seed(8191)

        # for fixed model_var_type's -> we assume the decoder variance is equal to the decoder
        self.fixed_model_var, self.fixed_model_logvar = {
            "fixed-large": (self.beta_sched.betas, torch.log(torch.cat([self.beta_sched.var_coefs[[1]], self.beta_sched.betas[1:]]))),
            "fixed-small": (self.beta_sched.var_coefs, self.beta_sched.var_coefs)
        }[self.model_var_type]

        # Clipping the Images dimension 
        self._clip = (lambda x: x.clamp(-1., 1.))

        return
    
    # INITIALIZERS -----------------------------------
    
    def _init_model(self):
        if self.config['network'] == 'unet':
            self.net = UNet(**self.config['unet'])
        else:
            raise 'Not Implimented - Network Architecture'
        return

    def forward(self, x):
        #configure the imputs to the network
        input_dict = self._extract_input_from_dataset(x)
        t = input_dict['t']

        #get noised inputs from the forward diffussion process
        x_t = self._sample_q_t(input_dict)

        if self.loss_type == "mse":
            print(x_t.size(), x_t.dtype)
            print(t.size(), t.dtype)
            
            model_out = self.net.forward(x_t, t)

        input_dict.update({'x_t':x_t})
        input_dict.update({'preds': model_out})

        return input_dict
    
    
    def loss(self, res_dict, x):
        #TODO i want to test out using the loss scalers - i think this incorrectly weights the losses currently 

        if self.loss_type == "dkl":
            loss = self._kl_loss(res_dict)

        elif self.loss_type == "mse":

            if self.vdm_type == "noise":
                #Learning to predict the source noise from a stanard gausian responsible for the samples x_t given x_0
                target = res_dict['noise']

            if self.vdm_type == "means":
                #Learning to predict mean of the noising distribution 
                target = self._sample_q_t_prev(res_dict)['mus']

            if self.vdm_type == "source":
                #Learning to predict the source image from the noised image
                target = res_dict['x_0']

            if self.vdm_type == "score":
                raise "Not Implimented - VDM type"

            #retreive the losses   
            # TODO not consitent 
            loss = torch.nn.functional.mse_loss(res_dict['preds'], target)

        elif self.loss_type == 'dkl':
            raise "Not Implimented - Loss Method"
            pass
        else:
            raise "Not Implimented - Loss Method"

        return {'loss':loss}
    

    def _kl_loss(self, input_dict):
        """
        Computing the KL based loss
        calculate L_t
        t = 0: negative log likelihood of decoder, -\log p(x_0 | x_1)
        t > 0: variational lower bound loss term, KL term
        """
        true_dist = self._sample_q_t_prev(input_dict) #{mus, vars, samples}
        model_mean, model_var, model_log_var = self._p_mean_var(input_dict)

        kl = normal_kl(true_dist['mus'], true_dist['log_vars'], model_mean, model_log_var)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2

        decoder_nll = discretized_gaussian_loglik(input_dict['x_0'], model_mean, log_scale=0.5 * model_log_var).neg()
        decoder_nll = flat_mean(decoder_nll) / math.log(2.)
        losses = torch.where(torch.to(kl.device) > 0, kl, decoder_nll) # if t == 0 use the pixel wise Nll
        return losses
    

    def _extract_input_from_dataset(self, x):
        """
        return 
        X_0 -> image to reconstruct
        t -> for each image, determine how much noise to apply for training on this sample
        X_T -> fully noised sample -> standard gaussian
        """
    
        return {
            "x_0": x,
            "t": torch.empty((x.shape[0],), dtype=torch.int64).random_(
                to=self.timesteps, generator=self.generator), #sample uniform from [0, timesteps] -> [batch_size]
            "noise": torch.randn(*x.size() ,generator=self.generator) #sample gaussian -> [batch_size, channels, im_w, im_h]
        }
        

    # ENCODER FUNCTIONS ---------------------------------------------------------

    def _sample_q_t(self, input_dict):
        """
        forward diffussion process
        requires:
            - alpha_bar
        q(x_t|x_0) = n(x_t; sqrt(alpha_bar)x_0, (1-alpha_bar)I)
        """
        data_dims = input_dict['x_0'].ndim
        x_0 = input_dict['x_0']
        noise = input_dict['noise']

        #get the distribution paramters for each timestep
        mean_coefs = self.beta_sched.get_sqrt_alpha_bars(input_dict['t'])
        stds = self.beta_sched.get_sqrt_one_minus_alpha_bars(input_dict['t'])

        # we must reshape these for the coeficients for the shape data
        mean_coefs = mean_coefs.reshape((-1,) + (1, ) * (data_dims-1)) #[batch_size, 1,1,1] assuming data dims 
        stds = stds.reshape((-1,) + (1, ) * (data_dims-1))

        #sample the encoder distrubtion using the reparameterization trick
        sample = mean_coefs * x_0 + stds * noise

        return sample 
    

    def _sample_q_t_prev(self, input_dict, sample=False):
        """
        q(x_t-1 | x_0, x_t)

        must compute 
            - mu-hat(x_t, x_0) 
            - beta_hat
        """
        x_0 = input_dict['x_0']
        x_t = input_dict['x_t']
        t = input_dict['t']
        data_dims = input_dict['x_0'].ndims

        #Compute means

        #Collect coeficients
        mu_coef_x0s, mu_coef_xts, var_coefs = self.beta_sched.get_posterior_coeficients(t-1.0)
   
        #reshape coefiecients for image dims
        mu_coef_x0s = mu_coef_x0s.reshape((-1,) + (1, ) * (data_dims-1)) #[batch_size, 1,1,1] assuming data dims 
        mu_coef_xts = mu_coef_xts.reshape((-1,) + (1, ) * (data_dims-1)) #[batch_size, 1,1,1] assuming data dims 
        var_coefs = var_coefs.reshape((-1,) + (1, ) * (data_dims-1)) #[batch_size, 1,1,1] assuming data dims 

        #means
        mus = mu_coef_x0s * x_0 + mu_coef_xts * x_t

        out_dict = {
            "mus":mus,
            "vars":var_coefs,
            "log_vars": torch.log(var_coefs)
        }
        
        #sample via reperameterization 
        if sample:
            noise = torch.randn(*x_0.size(), generator=self.generator)
            x_t_prev = mus + torch.sqrt(var_coefs) * noise 
            out_dict['samples'] = x_t_prev

        return out_dict
    
    
    
    # DECODER FUNCTIONS ---------------------------------------------------------

    def _p_mean_var(self, input_dict):
        """
        This function is used when we are using the KL loss methods 
        - used for getting the parameterization of the p guassian by the model 

        the model can be trained to predict different outcomes, 
        but the KL method depends on the paramterization of gaussians,
        we must return this paramterization 
        """

        model_out = self.net(input_dict['x_t'], input_dict['t']) 

        if self.model_var_type == 'learned':
            #TODO: this assumes the model must also output the variances and the means 
            raise "Not Implimented - KL loss method - learning variance"
        elif self.model_var_type in ['fixed-small', 'fixed-large']:
            # this gets the variance of the decdoer
            # Note: if we dont use KL loss these variances are only used in the generation processes
            model_var, model_log_var = self.fixed_model_logvar
            pass
        else:
            raise "Not Implimented - Model Variance Method"
        
        #Determine the model Means
        if self.vdm_type == 'mean':
            #the mean is provided directly from the output
            return model_out, model_var, model_log_var
        
        elif self.vdm_type == 'noise':
            #the model predict the nosie -> we must convert this to a prediction on x_0 then extract the implied mean 
            pred_x_0 = self._predict_x_0_form_noise(input_dict['x_t'], input_dict['t'], model_out)
            out_dict = self._sample_q_t_prev({
                'x_0':pred_x_0, 'x_t':input_dict['x_t'], 't':input_dict['t']
            })
            return out_dict['mus'], model_var, model_log_var

        elif self.vdm_type == 'source':
            #TODO do we need to clip this prediction
            #the model provides a prediction of x_0 -> from this we can infer the mean 
            out_dict = self._sample_q_t_prev({
                'x_0':model_out, 'x_t':input_dict['x_t'], 't':input_dict['t']
            })
            return out_dict['mus'], model_var, model_log_var
            
        elif self.vdm_type == "score":
            raise "Not Implimented"
        else:
            raise "Not Implimented"

    

    # X_0 PREDICTION HELPERS ---------------------------------------

    def _predict_x_0_form_noise(self, x_t, t, noise):
        """
        given the nosie - and x_t, this is the inreverse of the diffusion process
        """
        #TODO i dont understand this must go over
        coef1, coef2 = self.beta_sched.get_incerse_noising_coeficients(t)

        coef1 = coef1.reshape((-1,) + (1, ) * (x_t.ndims - 1))
        coef2 = coef2.reshape((-1,) + (1, ) * (x_t.ndims - 1))

        pred_x_0 = coef1 * x_t - coef2 * noise

        return pred_x_0
    

    # SAMPLING -------------------------------------------------------

    def p_sample(self):

        return 
