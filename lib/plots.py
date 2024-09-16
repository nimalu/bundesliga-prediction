import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from tabulate import tabulate


def cross_val_plot(df_matches, splits, labels, scores, color=None):
    scores = np.array(scores)
    sorting = np.argsort(-np.mean(scores, axis=1))
    labels = [labels[i] for i in sorting]
    scores = scores[sorting]
    if color is not None:
        color = np.array(color)[sorting]

    means = scores.mean(axis=1)
    stds = scores.std(axis=1)
    print(tabulate(zip(labels, means, stds), headers=["Model", "Score", "Std"], tablefmt="fancy_grid"))

    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    split_test_seasons = [df_matches.iloc[split[1][0]]["season"] for split in splits]

    if color is not None:
        axs[0].set_prop_cycle(cycler(color=color))
    axs[0].plot(scores.T, label=labels)
    axs[0].set_xticks(range(len(split_test_seasons)), split_test_seasons, rotation=90, ha="center")
    axs[0].set_xlabel("Test Saison")
    axs[0].set_ylabel("Punkte")
    axs[0].legend()

    if color is not None:
        bplot = axs[1].boxplot(scores.T, patch_artist=True)
        for patch, color in zip(bplot["boxes"], color):
            patch.set_facecolor(color)
    else:
        bplot = axs[1].boxplot(scores.T)

    axs[1].set_xlabel("Model")
    axs[1].set_xticks(range(1, len(labels) + 1), labels, rotation=45, ha="center")
    fig.tight_layout()
    return fig, axs
