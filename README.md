# Offline-RL

Offline reinforcement learning on fixed datasets, one idea at a time: behaviour cloning (version0), TD3+BC, which adds a critic and keeps the policy close to the data (version1), and conservative Q-learning (CQL), which instead makes the critic pessimistic about actions the data does not contain (version2). All three learn from the Minari *medium* datasets of Hopper and HalfCheetah and never interact with the environment except to evaluate the learned policy.

---

## Theory

Please note that the theory corresponding to the codes is hosted on this [page](https://parag1604.github.io/notes/Reinforcement-Learning/Offline-Reinforcement-Learning). The code follows its notation: a dataset $\mathcal{D}$ collected by a behaviour policy $\pi_\beta$, a learned policy $\pi_\theta$, critics $Q_\phi$ and discount $\gamma$. The online algorithms these build on (TD3, SAC) are in the [Actor-Critic](https://github.com/parag1604/Actor-Critic) repository.

## Python Environment Setup

Python >= 3.10, then:

```bash
pip install -r requirements.txt
```

A GPU is recommended (the results below used one); everything also runs on a CPU, only more slowly. `gymnasium[mujoco]` provides the Hopper-v5 and HalfCheetah-v5 environments used for evaluation, and `minari` loads the datasets.

## Data

The datasets come from [Minari](https://minari.farama.org/) (Farama Foundation), the maintained successor of D4RL:

| Task | Minari dataset ID | Transitions | Episodes | Mean episode return |
|---|---|---|---|---|
| Hopper-v5 | `mujoco/hopper/medium-v0` | 999,404 | 1,327 | 2818 |
| HalfCheetah-v5 | `mujoco/halfcheetah/medium-v0` | 1,000,000 | 1,000 | 12089 |

Each was collected by a single SAC policy trained part of the way (Stable-Baselines3); see the dataset pages on the Minari website. Download them (about 130 MB and 200 MB) with

```bash
minari download mujoco/hopper/medium-v0
minari download mujoco/halfcheetah/medium-v0
```

They go to `~/.minari/datasets` unless the environment variable `MINARI_DATASETS_PATH` points elsewhere.

**Normalized score.** As in D4RL, results are reported as

$$
\text{score} = 100 \cdot \frac{R - R_{\mathrm{random}}}{R_{\mathrm{expert}} - R_{\mathrm{random}}},
$$

so 0 is a uniformly random policy and 100 an expert. The Minari MuJoCo datasets ship no reference returns, so we measured them (see `common.py`): $R_{\mathrm{random}}$ is the mean return of 100 episodes of the uniform random policy (Hopper 17.0, HalfCheetah −275.3), and $R_{\mathrm{expert}}$ is the mean episode return of the matching expert dataset `mujoco/<env>/expert-v0` (Hopper 3857.8, HalfCheetah 16242.9). The datasets themselves score 72.9 (Hopper) and 74.9 (HalfCheetah). These references differ from D4RL's (different environment versions and expert policies), so the numbers here **cannot be compared with D4RL tables**. Minari's HalfCheetah "medium" policy in particular is much stronger than D4RL's.

## Running

`common.py` holds the shared pieces: the dataset as a GPU replay buffer (with state normalization by the dataset mean and standard deviation), the networks, evaluation and the normalized score. Every script has `--help`, `--task {hopper,halfcheetah}`, `--steps` (gradient steps), `--eval-every` and `--log runs/x.json`. Every 10k gradient steps the deterministic policy (the mean action) is evaluated for 10 episodes in the real environment.

`bash run_experiments.sh <python>` runs all three versions on both datasets (six runs in parallel on one GPU), and `python plot_results.py` draws the figures in `static/` and prints the numbers quoted below.

### version0: behaviour cloning

```bash
python version0.py --task hopper
```

Supervised learning of the dataset's actions, $\min_\theta \mathbb{E}_{(s,a)\sim\mathcal{D}} \lVert \pi_\theta(s) - a \rVert^2$. It ignores the rewards and can at best recover the behaviour policy, but on data from a single reasonably good policy that is already a strong baseline.

### version1: TD3+BC

```bash
python version1.py --task hopper
python version1.py --task hopper --alpha 0      # the actor reduces to BC
```

TD3 (twin critics, target policy smoothing, delayed actor updates) with one extra term in the actor objective (Fujimoto & Gu, 2021):

$$
\max_\theta \; \mathbb{E}_{(s,a)\sim\mathcal{D}} \Big[ \lambda \, Q_{\phi_1}(s, \pi_\theta(s)) - \big(\pi_\theta(s) - a\big)^2 \Big], \qquad \lambda = \frac{\alpha}{\frac{1}{B}\sum_{(s,a)} \lvert Q_{\phi_1}(s,a) \rvert}, \quad \alpha = 2.5 .
$$

The BC term keeps the policy where the critic was trained; $\lambda$ makes the trade-off independent of the reward scale.

### version2: CQL

```bash
python version2.py --task hopper
python version2.py --task hopper --cql-alpha 0   # plain offline SAC
```

Soft actor-critic whose critics are trained with the CQL($\mathcal{H}$) regularizer (Kumar et al., 2020):

$$
\alpha_{\mathrm{cql}} \, \mathbb{E}_{s\sim\mathcal{D}} \Big[ \log \sum_a \exp Q_\phi(s,a) - \mathbb{E}_{a\sim\mathcal{D}(\cdot\mid s)} Q_\phi(s,a) \Big] + \tfrac12 \mathbb{E}_{\mathcal{D}} \big[ (Q_\phi(s,a) - y)^2 \big], \qquad \alpha_{\mathrm{cql}} = 5 .
$$

The log-sum-exp over actions is estimated by importance sampling with 10 actions from each of three proposals (uniform, $\pi_\theta(\cdot\mid s)$ and $\pi_\theta(\cdot\mid s')$). It pushes $Q$ down wherever the policy (or anything else) would look for high values, and up on the actions in the data. The actor is a tanh-squashed Gaussian; for the first 100k steps (`--bc-steps`) it only imitates the data (MSE of its mean action, like version0) while the critic already trains with CQL, and then it switches to the SAC objective $\max_\theta \mathbb{E}[\min_i Q_{\phi_i}(s, a) - \tau \log \pi_\theta(a\mid s)]$ with a learned temperature $\tau$.

## Results

**Budget.** 300k gradient steps per run (batch 256, Adam), one seed, on one NVIDIA GB10 GPU with the six runs in parallel: about 20 minutes for BC, 30 for TD3+BC and 100 for CQL. The papers train for 1M steps, so these scores are not their final numbers. "Final" below is the mean normalized score of the last 5 evaluations (the last 50k steps, 50 episodes).

![Learning curves](static/learning_curves.png "Normalized score during training")

![Final scores](static/final_scores.png "Final normalized scores")

| Dataset (average score of the data) | v0: BC | v1: TD3+BC | v2: CQL |
|---|---|---|---|
| Hopper medium (72.9) | 91.2 | 92.4 | 92.3 |
| HalfCheetah medium (74.9) | 82.7 | 59.4 | 6.9 |

What the runs show:

- **BC beats the data.** The data was recorded from a *stochastic* SAC policy; BC learns the mean action, which is apparently better than the noisy actions that were actually taken (Hopper 91 vs 73, HalfCheetah 83 vs 75). On medium data, collected by a single fair policy, plain imitation is hard to beat.
- **Hopper:** all three end at about 92. TD3+BC and CQL are steadier than BC, whose evaluations occasionally drop to about 40 when the hopper falls.
- **HalfCheetah:** here using the critic *hurts*. TD3+BC, whose actor still has the BC term, ends at 59, below BC. CQL rises during the warm-up (52 at 100k steps, where BC was at 59) and collapses to about 2 as soon as the actor starts maximizing $Q$; it recovers only to 5–14. Minari's HalfCheetah medium policy is already strong (75% of expert) and its actions are very spread out (per-dimension standard deviation 0.7–0.8 on $[-1, 1]$), so the critic must rank actions whose effect on $Q \approx 1200$ is tiny, and its action-gradient is not reliable enough to improve on the mean action.

We did not get CQL to work on HalfCheetah within this budget. Shorter runs (40k–150k steps each) with $\alpha_{\mathrm{cql}} \in \{5, 10, 50\}$, rewards scaled by 0.1, or a smaller actor learning rate ($3\times10^{-5}$), all without the warm-up, never scored above 12; with a maximum-likelihood warm-up instead of the MSE one the actor barely learned during the warm-up (below 9). Without the warm-up, CQL on Hopper reaches the same score (92.4).
