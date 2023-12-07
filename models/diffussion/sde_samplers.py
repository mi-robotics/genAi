import torch
from scipy import integrate

class ReverseDiffusionPredictor():
  def __init__(self, sgm, probability_flow=False):
    super().__init__()
    self.sgm = sgm
    self.sde = self.sgm.sde

    # Compute the reverse SDE/ODE
    self.rsde = self.sde.reverse(sgm.score_function, probability_flow)
    self.score_fn = sgm.score_function
    return 

  def update(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean
    

class ODESSampler():
    def __init__(self, sgm, config):
        self.config = config
        self.use_denoise = config['sampler']['denoise']
        self.probability_flow = config['sampler']['probability_flow']
        self.eps = 1e-3
        self.rtol = 1e-5
        self.atol = 1e-5

        self.sgm = sgm
        self.score_fn = sgm.score_function
        self.rsde = sgm.sde.reverse(self.score_fn, self.probability_flow)

        if self.use_denoise:
            self.reverse_predictor = ReverseDiffusionPredictor(self.sgm, probability_flow=False)

        return
    
    def denoise(self, x):
        eps = torch.ones(x.size(0)) * self.eps
        x_update, x_mean = self.reverse_predictor.update(x, eps)
        return x_mean
    
    def ode_step(self, x, t, shape):
        x = torch.from_numpy(x.reshape(shape))
        t = torch.one(x.size(0)) * t
        drift, scale = self.rsde.forward(x,t)
        return drift.detach().cpu().numpy().reshape((-1,))
    
    def sample(self, z, shape, device='cpu'):
        """
        NOTE: Currently returns normalized samples
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = self.sde.prior_sampling(shape).to(device)
            else:
                x = z

            solution = integrate.solve_ivp(self.ode_step, 
                                      (self.sde.T, self.eps),
                                      x.detach().cpu().numpy().reshape((-1,)),
                                      rtol=self.rtol,
                                      atol=self.atol,
                                      method='RK45')
            
            nfe = solution.nfev
            x_0 = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if self.use_denoise:
                x = self.denoise(x)

            return x, nfe
    