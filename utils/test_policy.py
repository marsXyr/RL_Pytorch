import time
import joblib
import os.path as osp
import torch
import sys

sys.path.append('../')
from utils import core
from utils.logx import EpochLogger


def load_policy(fpath, number):

    # load the things!
    if fpath is None: return
    policy = torch.load(osp.join(fpath, 'models', 'actor'+str(number)+'.pkl'), map_location=lambda storage, loc: storage)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+str(number)+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, policy


def run_policy(env, policy, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    state, reward, done, ep_return, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)
        action = policy(torch.FloatTensor(state).unsqueeze(0))[0]
        state, reward, done, _ = env.step(action.detach().numpy()[0])
        ep_return += reward
        ep_len += 1

        if done or (ep_len == max_ep_len):
            logger.store(EpRet=ep_return, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_return, ep_len))
            state, reward, done, ep_return, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=5000)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--number', '-num', type=int)
    args = parser.parse_args()

    env, policy = load_policy(args.fpath, args.number)

    run_policy(env, policy, args.len, args.episodes, not(args.norender))
