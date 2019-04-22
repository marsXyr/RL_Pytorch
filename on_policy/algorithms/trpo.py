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
from gym.spaces import Box
from torch.nn.utils import vector_to_parameters, parameters_to_vector

EPS = 1e-8


class TRPO:
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
                                          self.gamma, self.lam, self.info_shapes)

        # set random seed
        self.seed = args.seed + 10000 * proc_id()
        self.env.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def _init_parameters(self):

        self.epochs = args.epochs
        self.max_ep_len = args.max_ep_len
        self.save_freq = args.save_freq
        self.steps_per_epoch = args.steps_per_epoch
        # replay buffer
        self.gamma = args.gamma
        self.lam = args.lam
        self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
        if isinstance(self.action_space, Box):
            self.info_shapes = {'old_mu': [self.action_space.shape[-1]], 'old_log_std': [self.action_space.shape[-1]]}
        else:
            self.info_shapes = {'old_logits': [self.action_space.n]}

        self.vf_lr = args.vf_lr
        self.train_v_iters = args.train_v_iters
        self.backtrack_coef = args.backtrack_coef
        self.damping_coef = args.damping_coef
        self.cg_iters = args.cg_iters
        self.backtrack_iters = args.backtrack_iters
        self.delta = args.delta

        # Save the parameters setting to config file
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

    def _init_nets(self):

        # initial actor and critic
        self.actor_critic = ActorCritic(state_dim=self.state_dim, action_space=self.action_space)

        # initial optim
        self.vf_optim = optim.Adam(self.actor_critic.value_function.parameters(), lr=self.vf_lr)
        # initial loss
        self.loss = nn.MSELoss()

        core.sync_all_params(self.actor_critic.parameters())

    def cg(self, Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = torch.zeros_like(b)
        # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        r = b
        p = b
        r_dot_old = torch.dot(r, r)
        for _ in range(self.cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (torch.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update(self):
        batchs = [core.to_tensor(x) for x in self.replay_buffer.get()]
        states, actions, advantages, returns, logp_olds = batchs[:-len(self.replay_buffer.sorted_info_keys)]
        policy_args = dict(zip(self.replay_buffer.sorted_info_keys, batchs[-len(self.replay_buffer.sorted_info_keys):]))

        # Main outputs from computation graph
        _, logp, _, _, d_kl, v = self.actor_critic(states, actions, **policy_args)

        # Prepare hessian func, gradient eval
        ratio = (logp - logp_olds).exp()     # pi(a|s) / pi_old(a|s)
        pi_l_old = -(ratio * advantages).mean()
        v_l_old = self.loss(v, returns)

        g = core.flat_grad(pi_l_old, self.actor_critic.policy.parameters(), retain_graph=True)
        g = core.to_tensor(mpi_avg(g.numpy()))
        pi_l_old = mpi_avg(pi_l_old.item())

        def Hx(x):
            hvp = core.hessian_vector_product(d_kl, self.actor_critic.policy, x)
            if self.damping_coef > 0:
                hvp += self.damping_coef * x
            return core.to_tensor(mpi_avg(hvp.numpy()))

        # Core calculations for TRPO
        x = self.cg(Hx, g)
        alpha = torch.sqrt(2 * self.delta / (torch.dot(x, Hx(x)) + EPS))
        old_params = parameters_to_vector(self.actor_critic.policy.parameters())

        def set_and_eval(step):
            vector_to_parameters(old_params - alpha * x * step, self.actor_critic.policy.parameters())
            _, logp, _, _, d_kl = self.actor_critic.policy(states, actions, **policy_args)
            ratio = (logp - logp_olds).exp()
            pi_loss = -(ratio * advantages).mean()
            return mpi_avg(d_kl.item()), mpi_avg(pi_loss.item())

        for j in range(self.backtrack_iters):
            kl, pi_l_new = set_and_eval(step=self.backtrack_coef**j)
            if (kl <= self.delta) and (pi_l_new <= pi_l_old):
                self.logger.log('Accepting new params at step %d of line search.' % j)
                self.logger.store(BacktrackIters=j)
                break

            if j == self.backtrack_iters - 1:
                self.logger.log('Line search failed! Keeping old params.')
                self.logger.store(BacktrackIters=j)
                kl, pi_l_new = set_and_eval(step=0.)
        
        # Value function updates
        for _ in range(self.train_v_iters):
            v = self.actor_critic.value_function(states)
            v_loss = self.loss(v, returns)

            # Value function gradient step
            self.vf_optim.zero_grad()
            v_loss.backward()
            core.average_gradients(self.vf_optim.param_groups)
            self.vf_optim.step()
        
        v = self.actor_critic.value_function(states)
        v_l_new = self.loss(v, returns)
        
        # Log changes from update
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
                          DeltaLossPi=(pi_l_new - pi_l_old), DeltaLossV=(v_l_new - v_l_old))
    
    def run(self):
        # Create dir for model saving
        if not osp.exists(fpath): os.mkdir(fpath)

        start_time = time.time()
        state, reward, done, ep_return, ep_len = self.env.reset(), 0, False, 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            self.actor_critic.eval()
            for t in range(self.local_steps_per_epoch):
                action, _, logp, info, _, v = self.actor_critic(core.to_tensor(state.reshape(1, -1)))

                # save and log
                self.replay_buffer.store(state, action.detach().numpy(), reward, v.item(), logp.detach().numpy(),
                                         core.values_as_sorted_list(info))
                self.logger.store(VVals=v)

                state, reward, done, _ = self.env.step(action.detach().numpy()[0])
                ep_return += reward
                ep_len += 1

                terminal = done or (ep_len == self.max_ep_len)
                if terminal or (t == self.local_steps_per_epoch - 1):
                    if not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = reward if done else \
                        self.actor_critic.value_function(core.to_tensor(state.reshape(1, -1))).item()
                    self.replay_buffer.finish_path(last_val)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_return, EpLen=ep_len)
                    state, reward, done, ep_return, ep_len = self.env.reset(), 0, False, 0, 0

            # Perform TRPO update!
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
            self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('BacktrackIters', average_only=True)
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
    parser.add_argument('--save_freq', type=int, default=5)

    """  algorithm  """
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--backtrack_coef', type=float, default=0.8)
    parser.add_argument('--damping_coef', type=float, default=0.1)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--backtrack_iters', type=int, default=10)

    """  others  """
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='trpo')
    parser.add_argument('--cpu', type=int, default=2)

    args = parser.parse_args()

    # run parallel code with mpi
    mpi_fork(args.cpu)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Save model path
    fpath = osp.join(logger_kwargs['output_dir'], 'models')

    trpo = TRPO()
    trpo.run()
