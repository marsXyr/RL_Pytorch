import numpy as np
import torch
import torch.nn as nn
import scipy.signal
from utils.mpi_tools import broadcast, mpi_avg
from torch.nn.utils import parameters_to_vector


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


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def sync_all_params(param, root=0):
    data = nn.utils.parameters_to_vector(param).detach().numpy()
    broadcast(data, root)
    nn.utils.vector_to_parameters(to_tensor(data), param)


def average_gradients(param_groups):
    for param_group in param_groups:
        for p in param_group['params']:
            if p.requires_grad:
                p.grad.data.copy_(to_tensor(mpi_avg(p.grad.data.numpy())))


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]


def flat_grad(f, param, **kwargs):
    return parameters_to_vector(torch.autograd.grad(f, param, **kwargs))


def hessian_vector_product(f, policy, x):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, policy.parameters(), create_graph=True)
    return flat_grad((g * x.detach()).sum(), policy.parameters(), retain_graph=True)



