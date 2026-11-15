"""
version2 - conservative Q-learning, CQL(H) (Kumar et al., 2020), on top of SAC.

New compared to version1: instead of constraining the *policy* to stay near
the data, CQL makes the *critic* pessimistic about actions outside the data.
Each critic minimizes its usual TD error plus a regularizer that pushes Q down
on actions drawn from a broad distribution and up on the dataset actions:

    alpha_cql * E_{s ~ D} [ log sum_a exp Q_phi(s, a)  -  E_{a ~ D(.|s)} Q_phi(s, a) ]
              + 1/2 E_{(s, a, r, s') ~ D} [ (Q_phi(s, a) - y)^2 ].

The log-sum-exp is a soft maximum over all actions, so the first term lowers
whichever actions Q currently rates highest, unless the data supports them.
For continuous actions the integral is estimated by importance sampling with
N actions from each of three proposals (uniform on [-1, 1]^|A|, pi(.|s) and
pi(.|s')), each Q value corrected by the log density of its proposal:

    log sum_a exp Q(s, a)  ~  logsumexp_j [ Q(s, a_j) - log mu(a_j) ]  (+ const).

The actor and the target are those of soft actor-critic (SAC): a tanh-squashed
Gaussian pi_theta, twin critics, a learned entropy temperature tau with target
entropy -|A|, and y = r + gamma (1 - done) min_i Q_target_i(s', a'),
a' ~ pi_theta(.|s') (no entropy term in the target, as in common CQL code).
For the first 100k steps (--bc-steps) the actor only imitates the data (MSE
of its mean action, as in version0) while the critic already trains with
CQL; then it switches to the SAC objective, maximizing the conservative Q.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Offline-Reinforcement-Learning
(conservative value estimates).

Run:  python version2.py
      python version2.py --cql-alpha 0    (no CQL: plain offline SAC)
      python version2.py --bc-steps 0     (no warm-up)
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (DEVICE, Critic, Logger, OfflineBuffer, evaluate,
                    make_parser, mlp, set_seed, soft_update)


class TanhGaussianActor(nn.Module):
    '''a = tanh(u), u ~ N(mu_theta(s), sigma_theta(s)^2), as in SAC.'''
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden_dim)

    def forward(self, s):
        '''A sampled action and its log-probability log pi_theta(a | s).'''
        mu, log_std = self.net(s).chunk(2, -1)
        std = log_std.clamp(-5, 2).exp()
        u = mu + std * torch.randn_like(mu)                   # reparameterized
        a = torch.tanh(u)
        log_prob = torch.distributions.Normal(mu, std).log_prob(u).sum(-1)
        log_prob -= (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, log_prob                     # the tanh change of variables

    def act(self, s):
        return torch.tanh(self.net(s).chunk(2, -1)[0])         # the mean action


def main():
    p = make_parser(__doc__.split('\n')[1], steps=300_000)
    p.add_argument('--cql-alpha', type=float, default=5.0)
    p.add_argument('--n-actions', type=int, default=10)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--polyak', type=float, default=0.005)
    p.add_argument('--policy-lr', type=float, default=1e-4)
    p.add_argument('--bc-steps', type=int, default=100_000,
                   help='behaviour-cloning warm-up of the actor')
    args = p.parse_args()
    set_seed(args.seed)
    D = OfflineBuffer(args.task)
    B, N, A = args.batch_size, args.n_actions, D.act_dim

    pi = TanhGaussianActor(D.obs_dim, A).to(DEVICE)
    Q = Critic(D.obs_dim, A, n_hidden=3).to(DEVICE)
    Q_target = copy.deepcopy(Q)
    log_tau = torch.zeros(1, requires_grad=True, device=DEVICE)  # entropy temp.
    pi_opt = torch.optim.Adam(pi.parameters(), lr=args.policy_lr)
    Q_opt = torch.optim.Adam(Q.parameters(), lr=3e-4)
    tau_opt = torch.optim.Adam([log_tau], lr=3e-4)
    log = Logger('cql', args, D)

    def q_rep(s_rep, a_rep):
        '''Both critics at N actions per state, shaped (B, N) each.'''
        q1, q2 = Q(s_rep, a_rep)
        return q1.view(B, N), q2.view(B, N)

    for step in range(1, args.steps + 1):
        s, a, r, s2, done = D.sample(B)

        # ---- actor and entropy temperature (SAC)
        if step <= args.bc_steps:     # warm-up: the mean action imitates D
            actor_loss = F.mse_loss(pi.act(s), a)
        else:                         # SAC: maximize E[Q - tau log pi_theta]
            a_pi, logp = pi(s)
            actor_loss = (log_tau.exp().detach() * logp
                          - torch.min(*Q(s, a_pi))).mean()
            tau_loss = -(log_tau * (logp.detach() - A)).mean()  # target -|A|
            tau_opt.zero_grad()
            tau_loss.backward()
            tau_opt.step()
        pi_opt.zero_grad()
        actor_loss.backward()
        pi_opt.step()

        # ---- critic: TD error
        with torch.no_grad():
            a2, _ = pi(s2)
            y = r + args.gamma * (1 - done) * torch.min(*Q_target(s2, a2))
        q1, q2 = Q(s, a)
        td_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        # ---- critic: the CQL(H) regularizer with 3 x N sampled actions
        s_rep = s.repeat_interleave(N, 0)
        with torch.no_grad():
            a_rand = torch.rand(B * N, A, device=DEVICE) * 2 - 1
            a_cur, logp_cur = pi(s_rep)
            a_next, logp_next = pi(s2.repeat_interleave(N, 0))
        log_unif = A * math.log(0.5)                 # log density of U[-1,1]^A
        q_rand, q_cur, q_next = (q_rep(s_rep, x)
                                 for x in (a_rand, a_cur, a_next))
        cql_loss = 0.0
        for i, q_data in enumerate((q1, q2)):
            cat = torch.cat([q_rand[i] - log_unif,
                             q_cur[i] - logp_cur.view(B, N),
                             q_next[i] - logp_next.view(B, N)], 1)
            cql_loss = cql_loss + (torch.logsumexp(cat, 1) - q_data).mean()
        Q_opt.zero_grad()
        (td_loss + args.cql_alpha * cql_loss).backward()
        Q_opt.step()
        soft_update(Q_target, Q, args.polyak)

        if step % args.eval_every == 0:
            ret, score = evaluate(pi, D, args.task, args.eval_episodes,
                                  args.seed)
            log.log(step, ret, score,
                    f'Q {q1.mean().item():7.1f}  cql {cql_loss.item():7.2f}  '
                    f'tau {log_tau.exp().item():.3f}')
    log.save()


if __name__ == '__main__':
    main()
