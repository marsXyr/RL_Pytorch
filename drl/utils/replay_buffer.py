import numpy as np
import torch

FloatTensor = torch.FloatTensor


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

    def size(self):
        if self.full:
            return self.buffer_size
        return self.pos

    def get_pos(self):
        return self.pos

    # Expects tuples of (state, next_state, action, reward, done)

    def add(self, datum):

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
        batch_inds = torch.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))

        return dict(states=self.states[batch_inds],
                    next_states=self.n_states[batch_inds],
                    actions=self.actions[batch_inds],
                    rewards=self.rewards[batch_inds],
                    dones=self.dones[batch_inds])

    def get_reward(self, start_pos, end_pos):

        tmp = 0
        if start_pos <= end_pos:
            for i in range(start_pos, end_pos):
                tmp += self.rewards[i]
        else:
            for i in range(start_pos, self.buffer_size):
                tmp += self.rewards[i]

            for i in range(end_pos):
                tmp += self.rewards[i]

        return tmp

    def repeat(self, start_pos, end_pos):

        if start_pos <= end_pos:
            for i in range(start_pos, end_pos):

                self.states[self.pos] = self.states[i].clone()
                self.n_states[self.pos] = self.n_states[i].clone()
                self.actions[self.pos] = self.actions[i].clone()
                self.rewards[self.pos] = self.rewards[i].clone()
                self.dones[self.pos] = self.dones[i].clone()

                self.pos += 1
                if self.pos == self.buffer_size:
                    self.full = True
                    self.pos = 0

        else:
            for i in range(start_pos, self.buffer_size):

                self.states[self.pos] = self.states[i].clone()
                self.n_states[self.pos] = self.n_states[i].clone()
                self.actions[self.pos] = self.actions[i].clone()
                self.rewards[self.pos] = self.rewards[i].clone()
                self.dones[self.pos] = self.dones[i].clone()

                self.pos += 1
                if self.pos == self.buffer_size:
                    self.full = True
                    self.pos = 0

            for i in range(end_pos):

                self.states[self.pos] = self.states[i].clone()
                self.n_states[self.pos] = self.n_states[i].clone()
                self.actions[self.pos] = self.actions[i].clone()
                self.rewards[self.pos] = self.rewards[i].clone()
                self.dones[self.pos] = self.dones[i].clone()

                self.pos += 1
                if self.pos == self.buffer_size:
                    self.full = True
                    self.pos = 0
