import torch 
import numpy as np
import matplotlib.pyplot as plt



def remove_mean(samples, n_particles, n_dimensions, eq_lattice_com=None):
    """Makes a configuration of many particle system mean-free.

    Parameters
    ----------
    samples : torch.Tensor
        Positions of n_particles in n_dimensions.

    Returns
    -------
    samples : torch.Tensor
        Mean-free positions of n_particles in n_dimensions.
    """
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True) + eq_lattice_com
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True) + eq_lattice_com
        samples = samples.reshape(*shape)
    return samples


class Coef:
    def __init__(self, coef, a=1.0, b=0.001):
        self.coef = coef
        self.a = a
        self.b = b
    
    def t(self, t):
        if self.coef == 't':
            return t
        elif self.coef == '1-t':
            return 1-t
        elif self.coef == 'cos(pi*t/2)':
            return torch.cos( 1/2 * np.pi * t)
        elif self.coef == 'sin(pi*t/2)':
            return torch.sin( 1/2 * np.pi * t)
        elif self.coef == 'sqrt(a*t*(1-t))':
            return torch.sqrt(self.a*t*(1-t))
        elif self.coef == 'sqrt(a*t*(1-t))+b':
            return torch.sqrt(self.a*t*(1-t)) + self.b
        elif self.coef == 'a*t*(1-t)':
            return self.a*t*(1-t)
        elif self.coef == 'a*sin(pi*t)':
            return self.a*torch.sin(np.pi*t)
        else:
            return torch.zeros_like(t) + float(self.coef)
        
    def dt(self, t):
        if self.coef == 't':
            return torch.ones_like(t)
        elif self.coef == '1-t':
            return -torch.ones_like(t)
        elif self.coef == 'cos(pi*t/2)':
            return -torch.sin( 1/2 * np.pi * t) * 1/2 * np.pi
        elif self.coef == 'sin(pi*t/2)':
            return torch.cos( 1/2 * np.pi * t) * 1/2 * np.pi
        elif self.coef == 'sqrt(a*t*(1-t))':
            return 1 / 2 / torch.sqrt(t*(1-t)) * self.a**0.5 * (1 - 2*t)
        elif self.coef == 'sqrt(a*t*(1-t))+b':
            return 1 / 2 / torch.sqrt(t*(1-t)) * self.a**0.5 * (1 - 2*t)
        elif self.coef == 'a*t*(1-t)':
            return self.a - 2 * t * self.a
        elif self.coef == 'a*sin(pi*t)':
            return self.a * np.pi * torch.cos(np.pi*t)
        else:
            return torch.zeros_like(t)    

    def tdt(self, t):
        if self.coef == 'sqrt(a*t*(1-t))': 
            return 1/2 * self.a * (1 - 2*t)
        else:
            return self.t(t) * self.dt(t)
