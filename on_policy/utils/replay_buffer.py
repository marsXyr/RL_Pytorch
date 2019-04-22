import numpy as np
from utils.mpi_tools import mpi_statistics_scalar
from on_policy.utils.core import combined_shape, discount_cumsum, keys_as_sorted_list, values_as_sorted_list


class ReplayBuffer:
    """
    A buffer for storing trajectories experienced by on-policy agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, buffer_size, state_dim, action_dim, gamma=0.99, lam=0.95, info_shapes=None):
        self.use_info = True if info_shapes is not None else False

        self.states = np.zeros(combined_shape(buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(buffer_size, action_dim), dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.logps = np.zeros(buffer_size, dtype=np.float32)
        if self.use_info:
            self.infos = {k: np.zeros([buffer_size] + list(v), dtype=np.float32) for k, v in info_shapes.items()}
            self.sorted_info_keys = keys_as_sorted_list(self.infos)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buffer_size

    def store(self, state, action, reward, value, logp, info=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # buffer has to have room so you can store
        assert self.ptr < self.max_size
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.logps[self.ptr] = logp
        if self.use_info:
            for i, k in enumerate(self.sorted_info_keys):
                self.infos[k][self.ptr] = info[i]
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_val)
        values = np.append(self.values[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.returns[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # buffer has to be full before you can get
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        mean, std = mpi_statistics_scalar(self.advantages)
        self.advantages = (self.advantages - mean) / std
        return [self.states, self.actions, self.advantages, self.returns, self.logps] + values_as_sorted_list(self.infos)\
            if self.use_info else [self.states, self.actions, self.advantages, self.returns, self.logps]
