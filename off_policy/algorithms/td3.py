import numpy as np, argparse, gym, time, random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import os.path as osp
from off_policy.utils import core
from off_policy.utils.model import Actor, CriticTD3
from off_policy.utils.replay_buffer import ReplayBuffer
from utils.logx import EpochLogger
from utils.run_utils import setup_logger_kwargs


class TD3(object):

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

        self.replay_buffer = ReplayBuffer(args.buffer_size, self.state_dim, self.action_dim)

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

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.gamma = args.gamma
        # Stddev for smoothing noise added to target policy
        self.target_noise = args.target_noise
        # Limit for absolute value of target policy
        self.noise_clip = args.noise_clip
        self.batch_size = args.batch_size
        # Policy will only be updated once every policy_decay times for each update of the Q-networks
        self.policy_delay = args.policy_delay

        # Whether use layer normalization in policy/value networks
        self.use_norm = args.use_norm
        # parameter for update policy/value networks
        self.tau = args.tau

        # Action noise added to policy at training time
        self.ouNoise = core.OUNoise(self.action_dim)
        self.gaNoise = core.GaussianNoise(self.action_dim)

        # Save the parameters setting to config file
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

    def _init_nets(self):
        # initial actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, self.use_norm)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.use_norm)
        self.critic = CriticTD3(self.state_dim, self.action_dim, self.use_norm)
        self.critic_target = CriticTD3(self.state_dim, self.action_dim, self.use_norm)

        # initial optim
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # initial loss
        self.loss = nn.MSELoss()

        # initial the actor target and critic target are the same as actor and critic
        core.hard_update(self.actor_target, self.actor)
        core.hard_update(self.critic_target, self.critic)

    def get_action(self, state, noise=None):
        action = core.to_numpy(self.actor(state))
        action = (action + noise.add()) if noise else action
        return np.clip(action, -self.action_limit, self.action_limit)

    def train(self, batch, policy_update=False):
        state_batch = batch['states']
        action_batch = batch['actions']
        next_state_batch = batch['next_states']
        reward_batch = batch['rewards']
        done_batch = batch['dones']

        # Target policy smoothing, by adding clipped noise to target actions
        noise = np.clip(np.random.normal(0, self.target_noise, size=(self.batch_size, self.action_dim)),
                        -self.noise_clip, self.noise_clip)
        next_action = self.actor_target(next_state_batch) + core.to_tensor(noise)
        next_action_clip = next_action.clamp(-self.action_limit, self.action_limit)

        q1_next, q2_next = self.critic_target(next_state_batch, next_action_clip)
        min_q_next = torch.min(q1_next, q2_next)
        # compute q_target and two q predict
        q_target = (reward_batch + self.gamma * (1 - done_batch.float()) * min_q_next).detach()
        q1_predict, q2_predict = self.critic(state_batch, action_batch)

        # critic update
        critic_loss = self.loss(q1_predict, q_target) + self.loss(q2_predict, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        if policy_update:
            # Delayed policy update
            q1, _ = self.critic(state_batch, self.actor(state_batch))
            actor_loss = -q1.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_optim.step()

            # actor/critic network soft update
            core.soft_update(self.actor_target, self.actor, self.tau)
            core.soft_update(self.critic_target, self.critic, self.tau)

            return critic_loss, actor_loss

        return critic_loss, 0.0

    def run(self):
        # Create dir for model saving
        if not osp.exists(fpath): os.mkdir(fpath)

        start_time = time.time()
        state, reward, done, ep_return, ep_len = self.env.reset(), 0, False, 0, 0
        total_steps = self.steps_per_epoch * self.epochs

        # Main loop: collect experience in env and update each epoch
        for t in range(total_steps):
            """
            Until start_steps have elapsed, randomly sample actions
            from a uniform distribution for better exploration. Afterwards, 
            use the learned policy (with some noise, via OUNoise).
            """
            if t > self.start_steps:
                action = self.get_action(core.to_tensor(state).unsqueeze(0), noise=self.ouNoise)
            else:
                action = self.env.action_space.sample()
            # Step the env
            next_state, reward, done, _ = self.env.step(action)
            ep_return += reward
            ep_len += 1

            """
            Ignore the "done" signal if it comes from hitting the time
            horizon (that is, when it's an artificial terminal signal
            that isn't based on the agent's state)
            """
            done = False if ep_len == self.max_ep_len else done
            done_bool = 0 if ep_len == self.max_ep_len else float(done)
            # Add the experience to the replay buffer
            self.replay_buffer.store((state, action, next_state, reward, done_bool))
            # Super critical, easy to overlook step: make sure to update most recent observation!
            state = next_state
            if done or (ep_len == self.max_ep_len):
                """
                Perform all TD3 updates at the end of the trajectory
                (in accordance with source code of TD3 published by original authors).
                """
                for i in range(ep_len):
                    batch = self.replay_buffer.sample(self.batch_size)
                    # Delayed policy update
                    policy_update = True if (i % self.policy_delay == 0) else False
                    critic_loss, actor_loss = self.train(batch, policy_update)
                    self.logger.store(ActorLoss=actor_loss, CriticLoss=critic_loss)
                    self.logger.store(EpRet=ep_return, EpLen=ep_len)

                state, reward, done, ep_return, ep_len = self.env.reset(), 0, False, 0, 0

            # End of epoch wrap-up
            if (t > 0) and (t % self.steps_per_epoch == 0):
                epoch = t // self.steps_per_epoch

                # Save env and model
                if (epoch % self.save_freq == 0) or (epoch == epoch - 1):
                    self.logger.save_state({'env': self.env}, epoch)
                    self.save_model(epoch)

                # Test the performance of the deterministic version of the agent.
                self.test_agent(10)

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

    def test_agent(self, epochs):
        for j in range(epochs):
            state, reward, done, ep_return, ep_len = self.test_env.reset(), 0, False, 0, 0
            while not (done or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                action = self.get_action(core.to_tensor(state).unsqueeze(0))
                state, reward, done, _ = self.test_env.step(action)
                ep_return += reward
                ep_len += 1

            self.logger.store(TestEpRet=ep_return, TestEpLen=ep_len)

    def save_model(self, epoch):
        torch.save(self.actor, osp.join(fpath, 'actor' + str(epoch) + '.pkl'))
        torch.save(self.critic, osp.join(fpath, 'critic' + str(epoch) + '.pkl'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """  env  """
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=10000)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=5)
    """  algorithm  """
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--policy_delay', type=int, default=2)
    parser.add_argument('--target_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    """  model  """
    parser.add_argument('--use_norm', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.99)
    """  others  """
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='td3')

    args = parser.parse_args()

    # logger_kwargs = dict{output_dir:..., exp_name:...}
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    # Save model path
    fpath = osp.join(logger_kwargs['output_dir'], 'models')

    td3 = TD3()
    td3.run()