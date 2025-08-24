import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from tabulate import tabulate

from lib import theme


def host_advantage(df_matches):
    diffs = (df_matches["home_goals"] - df_matches["away_goals"]).values
    host_wins = np.count_nonzero(diffs > 0)
    guest_wins = np.count_nonzero(diffs < 0)
    draws = np.count_nonzero(diffs == 0)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title("Home Advantage")
    labels = ["Home", "Draw", "Away"]
    sizes = [host_wins, guest_wins, draws]
    color = [
        theme.MATCH_RESULT["home_wins"],
        theme.MATCH_RESULT["away_wins"],
        theme.MATCH_RESULT["draw"],
    ]

    _, labels, value_labels = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1d%%",
        startangle=90,
        pctdistance=0.75,
        colors=color,
        textprops={"fontsize": 12},
        wedgeprops={"linewidth": 2, "edgecolor": "white", "width": 0.5},
    )
    for l in value_labels:
        l.set_color("white")
        l.set_fontweight("bold")

    for l, c in zip(labels, color):
        l.set_color(c)
        l.set_fontweight("bold")

    fig.tight_layout()


def scoreline_distribution(df_matches, top_k=10):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Distribution of Match Results")
    goals_encoded = [
        f"{int(r[0])}:{int(r[1])}"
        for r in df_matches[["home_goals", "away_goals"]].values
    ]
    results, counts = np.unique(goals_encoded, return_counts=True)
    sort = np.argsort(-counts)
    counts = counts[sort]
    results = results[sort]

    counts = counts[:top_k]
    results = results[:top_k]

    tendency = [1] * len(results)
    for i, res in enumerate(results):
        goals = res.split(":")
        if goals[0] > goals[1]:
            tendency[i] = 0
        elif goals[0] < goals[1]:
            tendency[i] = 2

    y_test_colors = [
        [
            theme.MATCH_RESULT["home_wins"],
            theme.MATCH_RESULT["draw"],
            theme.MATCH_RESULT["away_wins"],
        ][r]
        for r in tendency
    ]
    ax.bar(range(len(results)), counts, color=y_test_colors)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(results)), results, rotation=90)

    red_patch = mpatches.Patch(color=theme.MATCH_RESULT["away_wins"], label="Away wins")
    green_patch = mpatches.Patch(
        color=theme.MATCH_RESULT["home_wins"], label="Home wins"
    )
    gray_patch = mpatches.Patch(color=theme.MATCH_RESULT["draw"], label="Draw")
    ax.legend(handles=[red_patch, green_patch, gray_patch])


def goals_distribution(df_matches):
    all_goals = np.concat(
        [df_matches["home_goals"].values, df_matches["away_goals"].values]
    )
    values, counts = np.unique_counts(all_goals)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(values, counts)
    ax.set_xticks(range(len(values)))
    ax.set_xlabel("Goals")
    ax.set_ylabel("Frequency")


def cross_val_plot(df_matches, splits, labels, scores, color=None):
    scores = np.array(scores)
    sorting = np.argsort(-np.mean(scores, axis=1))
    labels = [labels[i] for i in sorting]
    scores = scores[sorting]
    if color is not None:
        color = np.array(color)[sorting]

    means = scores.mean(axis=1)
    stds = scores.std(axis=1)
    print(
        tabulate(
            zip(labels, means, stds),
            headers=["Model", "Score", "Std"],
            tablefmt="fancy_grid",
        )
    )

    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    split_test_seasons = [df_matches.iloc[split[1][0]]["season"] for split in splits]

    if color is not None:
        axs[0].set_prop_cycle(cycler(color=color))
    axs[0].plot(scores.T, label=labels)
    axs[0].set_xticks(
        range(len(split_test_seasons)), split_test_seasons, rotation=90, ha="center"
    )
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
