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
from utils.mpi_tools import mpi_fork, num_procs, mpi_avg, proc_id


class PPO:
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
        self.train_pi_iters = args.train_pi_iters
        self.train_v_iters = args.train_v_iters
        self.gamma = args.gamma
        self.lam = args.lam
        self.clip_ratio = args.clip_ratio
        self.target_kl = args.target_kl

        # Save the parameters setting to config file
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

    def _init_nets(self):

        # initial actor and critic
        self.actor_critic = ActorCritic(self.state_dim, self.action_space)
        # initial optim
        self.actor_optim = optim.Adam(self.actor_critic.policy.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.actor_critic.value_function.parameters(), lr=self.critic_lr)
        # initial loss
        self.loss = nn.MSELoss()

        core.sync_all_params(self.actor_critic.parameters())

    def update(self):
        states, actions, advantages, returns, logp_olds = [core.to_tensor(x) for x in self.replay_buffer.get()]

        # Training policy
        _, logp, _, _, _ = self.actor_critic.policy(states, actions)
        # a sample estimate for entropy
        entropy = (-logp).mean()

        ratio = (logp - logp_olds).exp()
        min_adv = torch.where(advantages > 0, (1 + self.clip_ratio) * advantages, (1 - self.clip_ratio) * advantages)
        pi_l_old = -(torch.min(ratio * advantages, min_adv)).mean()

        for i in range(self.train_pi_iters):
            # Output from policy function graph
            _, logp, _, _, _ = self.actor_critic.policy(states, actions)
            # PPO policy objective
            ratio = (logp - logp_olds).exp()
            min_adv = torch.where(advantages > 0, (1 + self.clip_ratio) * advantages, (1 - self.clip_ratio) * advantages)
            pi_loss = -(torch.min(ratio * advantages, min_adv)).mean()

            # Policy update
            self.actor_optim.zero_grad()
            pi_loss.backward()
            core.average_gradients(self.actor_optim.param_groups)
            self.actor_optim.step()

            # _, logp, _, _, _ = self.actor_critic.policy(states, actions)
            kl = (logp_olds - logp).mean()
            kl = mpi_avg(kl.item())
            if kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        self.logger.store(StopIter=i)

        # Training value function
        v = self.actor_critic.value_function(states)
        v_l_old = self.loss(v, returns)
        for _ in range(self.train_v_iters):
            # Output from value function graph
            v = self.actor_critic.value_function(states)
            # PPO value function objective
            v_loss = self.loss(v, returns)

            # Value function gradient step
            self.critic_optim.zero_grad()
            v_loss.backward()
            core.average_gradients(self.critic_optim.param_groups)
            self.critic_optim.step()

        # Log changes from update
        _, logp, _, _, _, v = self.actor_critic(states, actions)
        ratio = (logp - logp_olds).exp()
        min_adv = torch.where(advantages > 0, (1 + self.clip_ratio) * advantages,
                              (1 - self.clip_ratio) * advantages)
        pi_l_new = -(torch.min(ratio * advantages, min_adv)).mean()
        v_l_new = self.loss(v, returns)
        kl = (logp_olds - logp).mean()  # a sample estimate for KL-divergence
        clipped = (ratio > (1 + self.clip_ratio)) | (ratio < (1 - self.clip_ratio))
        cf = (clipped.float()).mean()
        self.logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=entropy,
            ClipFrac=cf,
            DeltaLossPi=(pi_l_new - pi_l_old),
            DeltaLossV=(v_l_new - v_l_old))

    def run(self):
        # Create dir for model saving
        if not osp.exists(fpath): os.mkdir(fpath)

        start_time = time.time()
        state, reward, done, ep_return, ep_len = self.env.reset(), 0, False, 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            self.actor_critic.eval()
            for t in range(self.local_steps_per_epoch):
                action, _, logp, _, _, v = self.actor_critic(core.to_tensor(state.reshape(1, -1)))

                # save and log
                self.replay_buffer.store(state, action.detach().numpy(), reward, v.item(), logp.detach().numpy())
                self.logger.store(VVals=v)

                state, reward, done, _ = self.env.step(action.detach().numpy()[0])
                ep_return += reward
                ep_len += 1

                terminal = done or (ep_len == self.max_ep_len)
                if terminal or (t == self.local_steps_per_epoch - 1):
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

            # Perform PPO update!
            self.actor_critic.train()
            self.update()

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
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
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time() - start_time)
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
    parser.add_argument('--train_pi_iters', type=int,default=80)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--save_freq', type=int, default=5)
    """  algorithm  """
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    """  others  """
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--cpu', type=int, default=2)

    args = parser.parse_args()

    # run parallel code with mpi
    mpi_fork(args.cpu)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Save model path
    fpath = osp.join(logger_kwargs['output_dir'], 'models')

    ppo = PPO()
    ppo.run()
