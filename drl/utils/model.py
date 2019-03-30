import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def initial_weights_(tensor):
    classname = tensor.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(tensor.weight, gain=1)
        nn.init.constant_(tensor.bias, 0)





class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, layer_norm=False):
        super(Actor, self).__init__()

        self.layer_norm = layer_norm
        # hidden layer dim
        l1_dim, l2_dim = 128, 128

        self.l1 = nn.Linear(state_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, action_dim)

        # use layer normalization
        if layer_norm:
            self.n1 = nn.LayerNorm(l1_dim)
            self.n2 = nn.LayerNorm(l2_dim)

        # Init
        self.apply(initial_weights_)

    def forward(self, state):

        if not self.layer_norm:
            out = torch.tanh(self.l1(state))
            out = torch.tanh(self.l2(out))
            out = torch.tanh(self.l3(out))
        else:
            out = torch.tanh(self.n1(self.l1(state)))
            out = torch.tanh(self.n2(self.l2(out)))
            out = torch.tanh(self.l3(out))

        return out


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, layer_norm=False):
        super(Critic, self).__init__()
        self.layer_norm = layer_norm

        l1_dim, l2_dim = 200, 300

        self.l1 = nn.Linear(state_dim + action_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(l1_dim)
            self.n2 = nn.LayerNorm(l2_dim)

        self.apply(initial_weights_)

    def forward(self, state, action):

        if not self.layer_norm:
            out = F.leaky_relu(self.l1(torch.cat([state, action], 1)))
            out = F.leaky_relu(self.l2(out))
            out = self.l3(out)
        else:
            out = F.leaky_relu(self.n1(self.l1(torch.cat([state, action], 1))))
            out = F.leaky_relu(self.n2(self.l2(out)))
            out = self.l3(out)

        return out


class Critic_V(nn.Module):
    def __init__(self, state_dim, layer_norm=False):
        super(Critic_V, self).__init__()
        self.layer_norm = layer_norm

        l1_dim, l2_dim = 200, 300

        self.l1 = nn.Linear(state_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(l1_dim)
            self.n2 = nn.LayerNorm(l2_dim)

        self.apply(initial_weights_)

    def forward(self, state):

        if not self.layer_norm:
            out = F.leaky_relu(self.l1(state))
            out = F.leaky_relu(self.l2(out))
            out = self.l3(out)
        else:
            out = F.leaky_relu(self.n1(self.l1(state)))
            out = F.leaky_relu(self.n2(self.l2(out)))
            out = self.l3(out)

        return out


class Critic_TD3(nn.Module):
    def __init__(self, state_dim, action_dim, layer_norm):
        super(Critic_TD3, self).__init__()
        self.layer_norm = layer_norm

        l1_dim, l2_dim = 200, 300

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(l1_dim)
            self.n2 = nn.LayerNorm(l2_dim)

        # Q2 architecture
        self.L1 = nn.Linear(state_dim + action_dim, l1_dim)
        self.L2 = nn.Linear(l1_dim, l2_dim)
        self.L3 = nn.Linear(l2_dim, 1)

        if layer_norm:
            self.N1 = nn.LayerNorm(l1_dim)
            self.N2 = nn.LayerNorm(l2_dim)

        self.apply(initial_weights_)

    def forward(self, state, action):

        if not self.layer_norm:
            # Q1 network output
            out1 = F.leaky_relu(self.l1(torch.cat([state, action], 1)))
            out1 = F.leaky_relu(self.l2(out1))
            out1 = self.l3(out1)
            # Q2 network output
            out2 = F.leaky_relu(self.L1(torch.cat([state, action], 1)))
            out2 = F.leaky_relu(self.L2(out2))
            out2 = self.L3(out2)

        else:
            # use layer normalization
            out1 = F.leaky_relu(self.n1(self.l1(torch.cat([state, action], 1))))
            out1 = F.leaky_relu(self.n2(self.l2(out1)))
            out1 = self.l3(out1)

            out2 = F.leaky_relu(self.N1(self.L1(torch.cat([state, action], 1))))
            out2 = F.leaky_relu(self.N2(self.L2(out2)))
            out2 = self.L3(out2)

        return out1, out2


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GaussianPolicy, self).__init__()

        # hidden layer dim
        l1_dim, l2_dim = 128, 128

        self.l1 = nn.Linear(state_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l_mean = nn.Linear(l2_dim, action_dim)
        self.l_log_std = nn.Linear(l2_dim, action_dim)

        # Init
        self.apply(initial_weights_)

    def forward(self, state):
        out = F.relu(self.l1(state))
        out = F.relu(self.l2(out))
        mean = self.l_mean(out)
        log_std = self.l_log_std(out).clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, x_t, mean, log_std




