"""
Draw the figures in static/ and print the numbers quoted in the README from
the curves in runs/ (written by run_experiments.sh).

"Final" = mean normalized score over the last 5 evaluations (the last 50k
gradient steps, 50 episodes), to smooth out the evaluation noise.

Run:  python plot_results.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS, OUT = 'runs', 'static'
TASKS = {'hopper': 'Hopper (mujoco/hopper/medium-v0)'}
VERSIONS = {'v0': ('v0: BC', '#0072B2'),
            'v1': ('v1: TD3+BC', '#D55E00'),
            'v2': ('v2: CQL', '#009E73')}


def final(scores, k=5):
    return float(np.mean(scores[-k:]))


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {(v, t): json.load(open(f'{RUNS}/{v}_{t}.json'))
            for v in VERSIONS for t in TASKS}

    # learning curves
    fig, ax0 = plt.subplots(figsize=(6.5, 4.6))
    axes = [ax0]
    for ax, (t, title) in zip(axes, TASKS.items()):
        for v, (label, c) in VERSIONS.items():
            d = data[(v, t)]
            ax.plot(np.array(d['steps']) / 1e3, d['scores'], color=c,
                    label=label)
        ax.axhline(data[('v0', t)]['dataset_score'], color='gray', ls='--',
                   label='dataset average')
        bc_steps = data[('v2', t)]['args'].get('bc_steps', 0)
        if bc_steps:
            ax.axvline(bc_steps / 1e3, color=VERSIONS['v2'][1], ls=':',
                       label='CQL: end of BC warm-up')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('gradient steps (thousands)')
        ax.set_ylabel('normalized score')
        ax.grid(alpha=0.3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3,
               fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(f'{OUT}/learning_curves.png', dpi=150)

    # final scores
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    w = 0.25
    for j, (v, (label, c)) in enumerate(VERSIONS.items()):
        vals = [final(data[(v, t)]['scores']) for t in TASKS]
        bars = ax.bar(np.arange(len(TASKS)) + (j - 1) * w, vals, w,
                      color=c, label=label)
        ax.bar_label(bars, fmt='%.0f', fontsize=8)
    for i, t in enumerate(TASKS):
        ax.hlines(data[('v0', t)]['dataset_score'], i - 1.6 * w, i + 1.6 * w,
                  color='gray', ls='--')
    ax.set_xticks(range(len(TASKS)), ['Hopper medium'])
    ax.set_ylabel('final normalized score')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8, loc='upper right', ncol=3)
    ax.set_title('Dashed: average of the dataset', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/final_scores.png', dpi=150)

    for t in TASKS:
        d0 = data[('v0', t)]
        print(f'{t}: dataset return {d0["dataset_return"]:.0f} '
              f'(score {d0["dataset_score"]:.1f})')
        for v, (label, _) in VERSIONS.items():
            d = data[(v, t)]
            print(f'  {label:12s} final score {final(d["scores"]):6.1f}  '
                  f'final return {final(d["returns"]):8.0f}  best eval '
                  f'{max(d["scores"]):6.1f}  ({d["minutes"]:.0f} min)')


if __name__ == '__main__':
    main()
