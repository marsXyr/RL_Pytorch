import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete
from torch.distributions.kl import kl_divergence


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, layer_norm=True, output_squeeze=False):
        super(MLP, self).__init__()
        self.layer_norm = layer_norm
        self.output_squeeze = output_squeeze
        l1_dim, l2_dim = 64, 64

        self.l1 = nn.Linear(in_dim, l1_dim); nn.init.zeros_(self.l1.bias)
        self.l2 = nn.Linear(l1_dim, l2_dim); nn.init.zeros_(self.l1.bias)
        self.l3 = nn.Linear(l2_dim, out_dim); nn.init.zeros_(self.l1.bias)

    def forward(self, x):
        out = torch.tanh(self.l1(x))
        out = torch.tanh(self.l2(out))
        out = torch.tanh(self.l3(out))

        return out.squeeze() if self.output_squeeze else out


class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CategoricalPolicy, self).__init__()

        self.logits = MLP(state_dim, action_dim)

    def forward(self, x, a=None, old_logits=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        logp = policy.log_prob(a).squeeze() if a is not None else None

        if old_logits is not None:
            old_policy = Categorical(logits=old_logits)
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None

        info = {'old_logits': logits.detach().numpy()}

        return pi, logp, logp_pi, info, d_kl


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GaussianPolicy, self).__init__()

        self.mu = MLP(state_dim, action_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x, a=None, old_log_std=None, old_mu=None):
        mu = self.mu(x)
        policy = Normal(mu, self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        logp = policy.log_prob(a).sum(dim=1) if a is not None else None

        if (old_mu is not None) or (old_log_std is not None):
            old_policy = Normal(old_mu, old_log_std.exp())
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None

        info = {'old_mu': np.squeeze(mu.detach().numpy()), 'old_log_std': self.log_std.detach().numpy()}

        return pi, logp, logp_pi, info, d_kl


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space, policy=None):
        super(ActorCritic, self).__init__()

        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(state_dim, action_space.shape[0])
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(state_dim, action_space.n)
        else:
            self.policy = policy(state_dim, action_space)

        self.value_function = MLP(in_dim=state_dim, out_dim=1, output_squeeze=True)

    def forward(self, x, a=None, **kwargs):
        pi, logp, logp_pi, info, d_kl = self.policy(x, a, **kwargs)
        v = self.value_function(x)

        return pi, logp, logp_pi, info, d_kl, v

