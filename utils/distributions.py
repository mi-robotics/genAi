import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple

@torch.jit.script
def normal_kl(mean1, logvar1, mean2, logvar2):
    diff_logvar = logvar1 - logvar2
    kl = (-1.0 - diff_logvar).add(
        (mean1 - mean2).pow(2) * torch.exp(-logvar2)).add(
        torch.exp(diff_logvar)).mul(0.5)
    return kl


@torch.jit.script
def approx_std_normal_cdf(x):
    """
    Reference:
    Page, E. “Approximations to the Cumulative Normal Function and Its Inverse for Use on a Pocket Calculator.”
     Applied Statistics 26.1 (1977): 75–76. Web.
    """
    return 0.5 * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

@torch.jit.script
def discretized_gaussian_loglik(
        x, means, log_scale, precision: float = 1./255,
        cutoff: Union[float, Tuple[float, float]] = (-0.999, 0.999), tol: float = 1e-12):
    if isinstance(cutoff, float):
        cutoff = (-cutoff, cutoff)
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    x_centered = x - means
    inv_stdv = torch.exp(-log_scale)
    upper = inv_stdv * (x_centered + precision)
    cdf_upper = torch.where(
        x > cutoff[1], torch.as_tensor(1, dtype=torch.float32, device=x.device), approx_std_normal_cdf(upper))
    lower = inv_stdv * (x_centered - precision)
    cdf_lower = torch.where(
        x < cutoff[0], torch.as_tensor(0, dtype=torch.float32, device=x.device), approx_std_normal_cdf(lower))
    log_probs = torch.log(torch.clamp(cdf_upper - cdf_lower - tol, min=0).add(tol))
    return log_probs
