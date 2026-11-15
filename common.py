"""
Pieces shared by every version: loading a Minari dataset into a replay
buffer, state normalization, evaluation and the normalized score.

Notation follows the lecture
(https://parag1604.github.io/notes/Reinforcement-Learning/Offline-Reinforcement-Learning):
a fixed dataset D = {(s, a, r, s')} collected by a behaviour policy pi_beta,
a learned policy pi_theta and critics Q_phi. No new data is ever collected;
the environment is used only to *evaluate* the learned policy.
"""
import argparse
import json
import os
import random
import time

import gymnasium as gym
import minari
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {'hopper': 'mujoco/hopper/medium-v0'}

# Reference returns for the normalized score
#     score = 100 * (R - R_random) / (R_expert - R_random).
# The Minari MuJoCo datasets ship no reference scores, so we measured them:
# R_random is the mean return of 100 episodes of the uniform random policy in
# the Gymnasium v5 environment, and R_expert is the mean episode return of the
# matching Minari expert-v0 dataset (mujoco/<env>/expert-v0).
# They differ from the old D4RL references (which used other environments
# and other expert policies), so the scores are not comparable with D4RL tables.
REF_RETURNS = {'hopper': (17.0, 3857.8)}


def normalized_score(task, ret):
    lo, hi = REF_RETURNS[task]
    return 100.0 * (ret - lo) / (hi - lo)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data
class OfflineBuffer:
    '''
    All transitions of the dataset, on the GPU. done = 1 only for a real
    termination (the hopper fell); an episode cut by the 1000-step time limit
    is truncated, not terminated, so we still bootstrap from its last state.
    '''
    def __init__(self, task, normalize=True):
        self.dataset = minari.load_dataset(DATASETS[task])
        obs, act, rew, nxt, done, returns = [], [], [], [], [], []
        for ep in self.dataset.iterate_episodes():
            obs.append(ep.observations[:-1])
            nxt.append(ep.observations[1:])
            act.append(ep.actions)
            rew.append(ep.rewards)
            done.append(ep.terminations)
            returns.append(ep.rewards.sum())
        obs, nxt = np.concatenate(obs), np.concatenate(nxt)
        # state normalization: s <- (s - mean) / std, statistics of the dataset
        self.mean = obs.mean(0) if normalize else np.zeros(obs.shape[1])
        self.std = obs.std(0) + 1e-3 if normalize else np.ones(obs.shape[1])
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32,
                                      device=DEVICE)
        self.s = t((obs - self.mean) / self.std)
        self.s_next = t((nxt - self.mean) / self.std)
        self.a = t(np.concatenate(act))
        self.r = t(np.concatenate(rew))
        self.done = t(np.concatenate(done))
        self.size = len(self.s)
        self.obs_dim, self.act_dim = self.s.shape[1], self.a.shape[1]
        self.dataset_return = float(np.mean(returns))
        print(f'{DATASETS[task]}: {self.size} transitions, '
              f'{len(returns)} episodes, mean return {self.dataset_return:.1f} '
              f'(normalized {normalized_score(task, self.dataset_return):.1f})')

    def sample(self, batch_size):
        i = torch.randint(0, self.size, (batch_size,), device=DEVICE)
        return self.s[i], self.a[i], self.r[i], self.s_next[i], self.done[i]


# ---------------------------------------------------------------- networks
def mlp(input_dim, output_dim, hidden_dim=256, n_hidden=2):
    layers, d = [], input_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
        d = hidden_dim
    return nn.Sequential(*layers, nn.Linear(d, output_dim))


class DeterministicActor(nn.Module):
    '''a = pi_theta(s) in [-1, 1]^|A| (all action spaces here are [-1, 1]).'''
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden_dim)

    def forward(self, s):
        return torch.tanh(self.net(s))

    def act(self, s):
        return self(s)


class Critic(nn.Module):
    '''Twin critics Q_phi1(s, a), Q_phi2(s, a) (clipped double Q, as in TD3).'''
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden_dim, n_hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden_dim, n_hidden)

    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


@torch.no_grad()
def soft_update(target, source, tau):
    '''Polyak averaging: target <- (1 - tau) target + tau source.'''
    for pt, p in zip(target.parameters(), source.parameters()):
        pt.mul_(1 - tau).add_(tau * p)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(policy, buffer, task, episodes=10, seed=0):
    '''Mean return of the (deterministic) policy in the real environment.'''
    env = buffer.dataset.recover_environment()      # Hopper-v5
    returns = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + 1000 * k)
        done, ret = False, 0.0
        while not done:
            s = torch.as_tensor((obs - buffer.mean) / buffer.std,
                                dtype=torch.float32, device=DEVICE)
            a = policy.act(s.unsqueeze(0))[0].cpu().numpy()
            obs, r, terminated, truncated, _ = env.step(a)
            ret += r
            done = terminated or truncated
        returns.append(ret)
    env.close()
    ret = float(np.mean(returns))
    return ret, normalized_score(task, ret)


class Logger:
    '''Keeps the evaluation curve, prints it and saves it as JSON.'''
    def __init__(self, name, args, buffer):
        self.name, self.args, self.t0 = name, args, time.time()
        self.data = {'algo': name, 'args': vars(args),
                     'dataset': DATASETS[args.task],
                     'dataset_return': buffer.dataset_return,
                     'dataset_score': normalized_score(args.task,
                                                       buffer.dataset_return),
                     'steps': [], 'returns': [], 'scores': []}

    def log(self, step, ret, score, extra=''):
        self.data['steps'].append(step)
        self.data['returns'].append(ret)
        self.data['scores'].append(score)
        print(f'[{self.name} {self.args.task}] step {step:7d}  return '
              f'{ret:8.1f}  score {score:6.1f}  {extra} '
              f'({(time.time() - self.t0) / 60:.1f} min)', flush=True)

    def save(self):
        path = self.args.log or f'runs/{self.name}_{self.args.task}.json'
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.data['minutes'] = (time.time() - self.t0) / 60
        with open(path, 'w') as f:
            json.dump(self.data, f)


def make_parser(description, steps):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--task', default='hopper', choices=list(DATASETS))
    p.add_argument('--steps', type=int, default=steps,
                   help='gradient steps')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=10_000)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log', default=None, help='JSON file for the curve')
    return p
