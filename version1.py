"""
version1 - TD3+BC (Fujimoto & Gu, 2021): off-policy RL plus a BC term.

New compared to version0: we use the rewards. Plain off-policy actor-critic
(TD3) on a fixed dataset fails, because the actor climbs Q_phi towards
actions that never appear in D, where Q_phi is an unchecked extrapolation
(and the bootstrapped targets r + gamma Q(s', pi(s')) feed those errors
back). TD3+BC keeps TD3's critic update unchanged and adds one term to the
actor objective that keeps pi_theta close to the data:

    max_theta  E_{(s, a) ~ D} [ lambda Q_phi1(s, pi_theta(s))
                                - (pi_theta(s) - a)^2 ],
    lambda = alpha / mean_{(s, a)} |Q_phi1(s, a)|      (alpha = 2.5).

lambda rescales Q so that the trade-off between "improve" and "stay close"
does not depend on the scale of the rewards; alpha = 0 recovers BC, a large
alpha recovers TD3. The rest is TD3:
  - twin critics, target y = r + gamma (1 - done) min_i Q_target_i(s', a'),
  - target policy smoothing a' = clip(pi_target(s') + clip(noise), -1, 1),
  - delayed actor and target updates (every 2 critic steps), Polyak averaging.
State normalization (common.py) is also part of the paper's recipe.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Offline-Reinforcement-Learning
(policy constraints).

Run:  python version1.py --task hopper
      python version1.py --task halfcheetah --alpha 0    (pure BC actor)
"""
import copy

import torch
import torch.nn.functional as F

from common import (DEVICE, Critic, DeterministicActor, Logger, OfflineBuffer,
                    evaluate, make_parser, set_seed, soft_update)


def main():
    p = make_parser(__doc__.split('\n')[1], steps=300_000)
    p.add_argument('--alpha', type=float, default=2.5)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--polyak', type=float, default=0.005)
    p.add_argument('--policy-noise', type=float, default=0.2)
    p.add_argument('--noise-clip', type=float, default=0.5)
    p.add_argument('--policy-freq', type=int, default=2)
    args = p.parse_args()
    set_seed(args.seed)
    D = OfflineBuffer(args.task)

    pi = DeterministicActor(D.obs_dim, D.act_dim).to(DEVICE)
    Q = Critic(D.obs_dim, D.act_dim).to(DEVICE)
    pi_target, Q_target = copy.deepcopy(pi), copy.deepcopy(Q)
    pi_opt = torch.optim.Adam(pi.parameters(), lr=3e-4)
    Q_opt = torch.optim.Adam(Q.parameters(), lr=3e-4)
    log = Logger('td3bc', args, D)

    for step in range(1, args.steps + 1):
        s, a, r, s2, done = D.sample(args.batch_size)

        # ---- critic: the TD3 target, computed only from dataset states s'
        with torch.no_grad():
            noise = (torch.randn_like(a) * args.policy_noise).clamp(
                -args.noise_clip, args.noise_clip)
            a2 = (pi_target(s2) + noise).clamp(-1, 1)
            y = r + args.gamma * (1 - done) * torch.min(*Q_target(s2, a2))
        q1, q2 = Q(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        Q_opt.zero_grad()
        critic_loss.backward()
        Q_opt.step()

        # ---- actor: lambda Q - BC, every policy_freq steps
        if step % args.policy_freq == 0:
            a_pi = pi(s)
            q = Q(s, a_pi)[0]
            lam = args.alpha / q.abs().mean().detach()
            bc = F.mse_loss(a_pi, a)
            actor_loss = -lam * q.mean() + bc
            pi_opt.zero_grad()
            actor_loss.backward()
            pi_opt.step()
            soft_update(pi_target, pi, args.polyak)
            soft_update(Q_target, Q, args.polyak)

        if step % args.eval_every == 0:
            ret, score = evaluate(pi, D, args.task, args.eval_episodes,
                                  args.seed)
            log.log(step, ret, score,
                    f'Q {q1.mean().item():7.1f}  bc {bc.item():.4f}')
    log.save()


if __name__ == '__main__':
    main()
