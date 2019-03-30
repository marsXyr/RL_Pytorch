import numpy as np, argparse, gym, time, random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import os.path as osp
from drl.utils import core
from drl.utils.model import GaussianPolicy, Critic_TD3, Critic_V
from drl.utils.logx import EpochLogger
from drl.utils.run_utils import setup_logger_kwargs


class SAC(object):

    def __init__(self):

        # train env, which is used in train process
        self.env = gym.make(args.env).unwrapped
        # test env, which is used in eval process
        self.test_env = gym.make(args.env).unwrapped
        # env information
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        # action_limit for clamping: critically, assume all dimensions share the same bound
        self.action_limit = self.env.action_space.high[0]

        self.replay_buffer = core.ReplayBuffer(args.buffer_size)

        self._init_parameters()
        self._init_nets()

        # set random seed
        self.env.seed(args.seed)
        self.test_env.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    def _init_parameters(self):

        self.epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.start_steps = args.start_steps
        # Maximum length of trajectory / episode / rollout
        self.max_ep_len = args.max_ep_len
        # How often to save the current policy and value function and env
        self.save_freq = args.save_freq

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        # Entropy regularization coefficient
        self.alpha = args.alpha

        # Whether use layer normalization in policy/value networks
        self.use_norm = args.use_norm
        # parameter for update policy/value networks
        self.tau = args.tau

        # Save the parameters setting to config file
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

    def _init_nets(self):
        # initial one policy(P) net, two q(Q) nets, one v(V) net and one v_target_net(V_target)
        self.P = GaussianPolicy(self.state_dim, self.action_dim)
        self.Q = Critic_TD3(self.state_dim, self.action_dim, self.use_norm)
        self.V = Critic_V(self.state_dim, self.use_norm)
        self.V_target = Critic_V(self.state_dim, self.use_norm)

        # initial optim
        self.p_optim = optim.Adam(self.P.parameters(), lr=args.actor_lr)
        self.q_optim = optim.Adam(self.Q.parameters(), lr=args.critic_lr)
        self.v_optim = optim.Adam(self.V.parameters(), lr=args.critic_lr)

        # initial loss
        self.loss = nn.MSELoss()

        core.hard_update(self.V_target, self.V)

    def get_action(self, state, is_determinstic):
        if is_determinstic:
            _, _, _, action, _ = self.P.sample(state)
            action = torch.tanh(action)
        else:
            action, _, _, _, _ = self.P.sample(state)
        action = core.to_numpy(action.detach())
        return np.clip(action, -self.action_limit, self.action_limit)

    def add_experience(self, state, action, next_state, reward, done):
        reward = core.to_tensor(np.array([reward])).unsqueeze(0)
        done = core.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
        action = core.to_tensor(action)
        self.replay_buffer.push(state, action, next_state, reward, done)

    def train(self, batch):
        with torch.no_grad():
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            next_state_batch = torch.cat(batch.next_state)
            reward_batch = torch.cat(batch.reward)
            done_batch = torch.cat(batch.done)
        # Compute q_target
        q1, q2 = self.Q(state_batch, action_batch)
        v_next_target = self.V_target(next_state_batch)
        q_target = reward_batch + self.gamma * (1 - done_batch.float()) * v_next_target.detach()

        # Compute v_target
        sample_actions, log_prob, _, mean, log_std = self.P.sample(state_batch)
        q1_new, q2_new = self.Q(state_batch, sample_actions)
        min_q_new = torch.min(q1_new, q2_new)
        v_target = min_q_new - (self.alpha * log_prob)
        v = self.V(state_batch)

        # q network update
        self.q_optim.zero_grad()
        q_loss = self.loss(q1, q_target) + self.loss(q2, q_target)
        q_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.Q.parameters(), 10)
        self.q_optim.step()

        # v network update
        self.v_optim.zero_grad()
        v_loss = self.loss(v, v_target.detach())
        v_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.V.parameters(), 10)
        self.v_optim.step()

        """
        Reparameterization trick is used to get a low variance estimator
        """
        # policy network update
        self.p_optim.zero_grad()
        mean_loss = 0.001 * mean.pow(2).mean()
        std_loss = 0.001 * log_std.pow(2).mean()
        p_loss = (self.alpha * log_prob - min_q_new).mean() + mean_loss + std_loss
        p_loss.backward()
        nn.utils.clip_grad_norm_(self.P.parameters(), 10)
        self.p_optim.step()

        core.soft_update(self.V_target, self.V, self.tau)

        return q_loss, p_loss

    def run(self):
        # Create dir for model saving
        if not osp.exists(fpath): os.mkdir(fpath)

        start_time = time.time()
        state, reward, done, ep_return, ep_len = self.env.reset(), 0, False, 0, 0
        # Add the first(batch) dimension (None, state_dim)
        state = core.to_tensor(state).unsqueeze(0)
        total_steps = self.steps_per_epoch * self.epochs

        # main loop: collect experience in env and update each epoch
        for t in range(total_steps):
            """
            Until start_steps have elapsed, randomly sample actions
            from a uniform distribution for better exploration. Afterwards, 
            use the learned policy (with some noise, via OUNoise).
            """
            if t > self.start_steps:
                action = self.get_action(state, is_determinstic=False)
            else:
                action = self.env.action_space.sample()
                action = action[np.newaxis, :]
            # Step the env
            next_state, reward, done, _ = self.env.step(action.flatten())
            next_state = core.to_tensor(next_state).unsqueeze(0)
            ep_return += reward
            ep_len += 1

            """
            Ignore the "done" signal if it comes from hitting the time
            horizon (that is, when it's an artificial terminal signal
            that isn't based on the agent's state)
            """
            done = False if ep_len == self.max_ep_len else done
            # Add the experience to the replay buffer
            self.add_experience(state, action, next_state, reward, done)
            # Super critical, easy to overlook step: make sure to update most recent observation!
            state = next_state
            if done or (ep_len == self.max_ep_len):
                """
                Perform all SAC updates at the end of the trajectory
                (in accordance with source code of TD3 published by original authors).
                """
                for i in range(ep_len):
                    transitions = self.replay_buffer.sample(self.batch_size)
                    batch = core.Transition(*zip(*transitions))
                    critic_loss, actor_loss = self.train(batch)
                    self.logger.store(ActorLoss=actor_loss, CriticLoss=critic_loss)
                    self.logger.store(EpRet=ep_return, EpLen=ep_len)

                state, reward, done, ep_return, ep_len = self.env.reset(), 0, False, 0, 0
                state = core.to_tensor(state).unsqueeze(0)

            # End of epoch wrap-up
            if (t > 0) and (t % self.steps_per_epoch == 0):
                epoch = t // self.steps_per_epoch

                # Save env and model
                if (epoch % self.save_freq == 0) or (epoch == epoch - 1):
                    self.logger.save_state({'env': self.env}, epoch)
                    self.save_model(epoch)

                # Test the performance of the deterministic version of the agent.
                self.eval(10)

                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('ActorLoss', average_only=True)
                self.logger.log_tabular('CriticLoss', average_only=True)
                self.logger.log_tabular('Time', time.time() - start_time)
                self.logger.dump_tabular()

    def eval(self, epochs):
        for j in range(epochs):
            state, reward, done, ep_return, ep_len = self.test_env.reset(), 0, False, 0, 0
            state = core.to_tensor(state).unsqueeze(0)
            while not (done or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                action = self.get_action(state, is_determinstic=True)
                state, reward, done, _ = self.test_env.step(action.flatten())
                state = core.to_tensor(state).unsqueeze(0)
                ep_return += reward
                ep_len += 1

            self.logger.store(TestEpRet=ep_return, TestEpLen=ep_len)

    def save_model(self, epoch):
        torch.save(self.P, osp.join(fpath, 'actor' + str(epoch) + '.pkl'))
        torch.save(self.V, osp.join(fpath, 'critic' + str(epoch) + '.pkl'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """  env  """
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=10000)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=5)
    """  algorithm  """
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    """  model  """
    parser.add_argument('--use_norm', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.99)
    """  others  """
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='sac')

    args = parser.parse_args()

    # logger_kwargs = dict{output_dir:..., exp_name:...}
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    # Save model path
    fpath = osp.join(logger_kwargs['output_dir'], 'models')

    sac = SAC()
    sac.run()


