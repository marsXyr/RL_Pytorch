import torch
import numpy as np


class OUNoise:
    """
    Ornstein-Uhnlenbeck process
    """
    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def add(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class GaussianNoise:
    """
    Simple Gaussian noise
    """
    def __init__(self, action_dim, sigma=0.2):
        self.action_dim = action_dim
        self.sigma = sigma

    def add(self):
        return np.random.normal(scale=self.sigma, size=self.action_dim)


def to_numpy(var):
    return var.data.numpy()


def to_tensor(x):
    x = np.array(x, dtype=np.float64).tolist()
    return torch.FloatTensor(x)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

