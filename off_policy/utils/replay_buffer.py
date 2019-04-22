import torch
import numpy as np


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor


class ReplayBuffer():

    def __init__(self, buffer_size, state_dim, action_dim):

        # params
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        self.states = torch.zeros(self.buffer_size, self.state_dim)
        self.actions = torch.zeros(self.buffer_size, self.action_dim)
        self.n_states = torch.zeros(self.buffer_size, self.state_dim)
        self.rewards = torch.zeros(self.buffer_size, 1)
        self.dones = torch.zeros(self.buffer_size, 1)

    # Expects tuples of (state, next_state, action, reward, done)
    def store(self, datum):

        state, action, n_state, reward, done = datum

        self.states[self.pos] = FloatTensor(state)
        self.actions[self.pos] = FloatTensor(action)
        self.n_states[self.pos] = FloatTensor(n_state)
        self.rewards[self.pos] = FloatTensor([reward])
        self.dones[self.pos] = FloatTensor([done])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):

        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = LongTensor(np.random.randint(0, upper_bound, size=batch_size))

        return dict(states=self.states[batch_inds],
                    next_states=self.n_states[batch_inds],
                    actions=self.actions[batch_inds],
                    rewards=self.rewards[batch_inds],
                    dones=self.dones[batch_inds])

