"""
version0 - behaviour cloning (BC): supervised learning on the dataset.

The simplest offline method ignores rewards altogether and imitates the
behaviour policy pi_beta that collected the data:

    min_theta  E_{(s, a) ~ D} || pi_theta(s) - a ||^2

(a deterministic actor with mean squared error, i.e. the maximum-likelihood
fit of a Gaussian policy with fixed variance). BC can only be as good as the
data on average, but on "medium" data, collected by one fixed, reasonably
good policy, that is already a strong baseline: the versions that follow use
the rewards to try to do better than pi_beta without leaving its support.

States are normalized with the dataset mean and standard deviation (common.py),
which helps every version here.

Lecture: https://parag1604.github.io/notes/Reinforcement-Learning/Offline-Reinforcement-Learning

Run:  python version0.py
"""
import torch
import torch.nn.functional as F

from common import (DEVICE, DeterministicActor, Logger, OfflineBuffer,
                    evaluate, make_parser, set_seed)


def main():
    args = make_parser(__doc__.split('\n')[1], steps=100_000).parse_args()
    set_seed(args.seed)
    D = OfflineBuffer(args.task)
    pi = DeterministicActor(D.obs_dim, D.act_dim).to(DEVICE)
    opt = torch.optim.Adam(pi.parameters(), lr=3e-4)
    log = Logger('bc', args, D)

    for step in range(1, args.steps + 1):
        s, a, _, _, _ = D.sample(args.batch_size)
        loss = F.mse_loss(pi(s), a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0:
            ret, score = evaluate(pi, D, args.task, args.eval_episodes,
                                  args.seed)
            log.log(step, ret, score, f'bc_loss {loss.item():.4f}')
    log.save()


if __name__ == '__main__':
    main()
