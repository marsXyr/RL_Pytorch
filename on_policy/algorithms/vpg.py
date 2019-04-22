import numpy as np
import argparse, gym, time, os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from on_policy.utils import core
from on_policy.utils.model import ActorCritic
from on_policy.utils.replay_buffer import ReplayBuffer
from utils.logx import EpochLogger
from utils.run_utils import setup_logger_kwargs
from utils.mpi_tools import mpi_fork, num_procs, proc_id


# Vanilla Policy Gradient (with GAE-Lambda for advantage estimation)
class VPG:
    def __init__(self):

        # train env, which is used in train process
        self.env = gym.make(args.env).unwrapped

        # env information
        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_dim = self.state_space.shape[0]

        self._init_parameters()
        self._init_nets()

        self.replay_buffer = ReplayBuffer(self.local_steps_per_epoch, self.state_space.shape, self.action_space.shape,
                                          self.gamma, self.lam)

        # set random seed
        seed = args.seed + 10000 * proc_id()
        self.env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _init_parameters(self):
        self.epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
        self.max_ep_len = args.max_ep_len
        self.save_freq = args.save_freq

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.train_v_iters = args.train_v_iters
        self.gamma = args.gamma
        self.lam = args.lam

        # Save the parameters setting to config file
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

    def _init_nets(self):

        # initial ActorCritic
        self.actor_critic = ActorCritic(self.state_dim, self.action_space)
        # initial optim
        self.actor_optim = optim.Adam(self.actor_critic.policy.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.actor_critic.value_function.parameters(), lr=self.critic_lr)
        # initial loss
        self.loss = nn.MSELoss()

        core.sync_all_params(self.actor_critic.parameters())

    def update(self):
        states, actions, advantages, returns, logps = [core.to_tensor(x) for x in self.replay_buffer.get()]

        # Policy gradient step
        _, logp, _, _, _ = self.actor_critic.policy(states, actions)
        # a sample estimate for entropy
        entropy = (-logp).mean()

        # Policy update
        self.actor_optim.zero_grad()
        # VPG policy objective
        policy_loss = -(logp * advantages).mean()
        policy_loss.backward()
        self.actor_optim.step()

        # value function update
        v_predict = self.actor_critic.value_function(states)
        v_loss_old = self.loss(v_predict, returns)
        for _ in range(self.train_v_iters):
            # Output from value function graph
            v_predict = self.actor_critic.value_function(states)
            # VPG value objective
            v_loss = self.loss(v_predict, returns)
            # Value function gradient step
            self.critic_optim.zero_grad()
            v_loss.backward()
            core.average_gradients(self.critic_optim.param_groups)
            self.critic_optim.step()

        # Log changes from update
        _, logp, _, _, _, v = self.actor_critic.forward(states, actions)
        policy_loss_new = -(logp * advantages).mean()
        v_loss_new = self.loss(v, returns)
        # A sample estimate for KL-divergence
        kl = (logps - logp).mean()
        self.logger.store(LossPi=policy_loss, LossV=v_loss_old, KL=kl, Entropy=entropy,
                          DeltaLossPi=(policy_loss_new - policy_loss), DeltaLossV=(v_loss_new - v_loss_old))

    def run(self):
        # Create dir for model saving
        if not osp.exists(fpath): os.mkdir(fpath)

        start_time = time.time()
        state, reward, done, ep_return, ep_len = self.env.reset(), 0, False, 0, 0
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            self.actor_critic.eval()
            for t in range(self.local_steps_per_epoch):
                action, _, logp, _, _, v = self.actor_critic.forward(core.to_tensor(state.reshape(1, -1)))
                # save and log
                self.replay_buffer.store(state, action.detach().numpy(), reward, v.item(), logp.detach().numpy())
                self.logger.store(VVals=v)

                state, reward, done, _ = self.env.step(action.detach().numpy()[0])
                ep_return += reward
                ep_len += 1

                terminal = done or (ep_len == self.max_ep_len)
                if terminal or (t == self.local_steps_per_epoch-1):
                    if not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = reward if done else \
                        self.actor_critic.value_function(core.to_tensor(state.reshape(1, -1))).item()
                    self.replay_buffer.finish_path(last_val)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_return, EpLen=ep_len)
                    state, reward, done, ep_return, ep_len = self.env.reset(), 0, False, 0, 0

            # Perform VPG update!
            self.actor_critic.train()
            self.update()

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                self.logger.save_state({'env': self.env}, None)
                self.save_model(epoch)

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()

    def save_model(self, epoch):
        torch.save(self.actor_critic, osp.join(fpath, 'actor_critic'+str(epoch)+'.pkl'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """  env  """
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--save_freq', type=int, default=5)
    """  algorithm  """
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    """  others  """
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--cpu', type=int, default=2)

    args = parser.parse_args()
    # run parallel code with mpi
    mpi_fork(args.cpu)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Save model path
    fpath = osp.join(logger_kwargs['output_dir'], 'models')

    vpg = VPG()
    vpg.run()