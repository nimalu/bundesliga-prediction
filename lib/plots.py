import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lib import scores, theme


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
    for label in value_labels:
        label.set_color("white")
        label.set_fontweight("bold")

    for label, c in zip(labels, color, strict=False):
        label.set_color(c)
        label.set_fontweight("bold")

    fig.tight_layout()


def scoreline_distribution(
    df_matches, top_k=10, goals_col=("home_goals", "away_goals")
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Distribution of Match Results")
    goals_encoded = [
        f"{int(r[0])}:{int(r[1])}"
        for r in df_matches[[goals_col[0], goals_col[1]]].values
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


def join_distribution_heatmap(probabilities):
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        probabilities,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Distribution"},
    )
    plt.xlabel("Away Goals")
    plt.ylabel("Home Goals")
    plt.title("Joint Distribution of Scorelines")


def expected_rewards_heatmap(probabilities):
    kicktipp_score_matrices = scores.generate_kicktipp_score_matrices(
        max_goals=probabilities.shape[1] - 1
    )
    rewards = np.einsum("ij,ijkl->kl", probabilities, kicktipp_score_matrices)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        rewards,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Rewards"},
    )
    plt.xlabel("Away Goals")
    plt.ylabel("Home Goals")
    plt.title("Expected Rewards for Scorelines")


def plot_team_strenghts(team_params):
    plt.figure(figsize=(10, 5))
    plt.scatter(team_params["attack"], team_params["defense"], alpha=0.7)

    for _i, row in team_params.iterrows():
        plt.annotate(
            row["team"],
            (row["attack"], row["defense"]),
            xytext=(5, -3),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Attack Strength")
    plt.ylabel("Defense Strength")
    plt.title("Team Attack vs Defense Parameters")
    plt.grid(True, alpha=0.3)


def kicktipp_score_summary(df_predictions):
    fig, axs = plt.subplots(
        ncols=2,
        figsize=(12, 5),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 3], "wspace": 0.25},
        dpi=500,
    )

    ax = axs[0]
    team_host_scores = (
        df_predictions[["home", "score"]].groupby(["home"]).sum().sort_index()
    )
    team_guest_scores = (
        df_predictions[["away", "score"]].groupby(["away"]).sum().sort_index()
    )
    team_scores = team_host_scores["score"].values + team_guest_scores["score"].values
    sort = np.argsort(team_scores)
    team_scores = team_scores[sort].astype(int)
    team_names = team_host_scores.index.values
    team_names = team_names[sort]

    color = [theme.GRAY] * len(team_names)
    color = np.array(color)
    ax.invert_xaxis()
    ax.barh(team_names, team_scores, color=color)
    ax.set_frame_on(False)
    ax.set_xlabel("Punkte")
    ax.set_xticks([0, 25, 50])

    ax = axs[1]
    ax.set_xlim(0, 34 + 1)
    ax.set_ylim(-1, 18)
    ax.set_xlabel("Spieltag")
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks(range(1, 36, 11))
    ax.set_frame_on(False)

    for i, team in enumerate(team_names):
        ax.text(-2.5, i, team, horizontalalignment="center", verticalalignment="center")
        for matchday in range(1, 34 + 1):
            s = df_predictions.loc[
                (df_predictions["match_day"] == matchday)
                & ((df_predictions["home"] == team) | (df_predictions["away"] == team)),
                "score",
            ].values[0]
            if s == 4:
                color = theme.GRASS_D
            elif s == 3:
                color = theme.GRASS
            elif s == 2:
                color = theme.GRASS_B
            else:
                color = "lightgray"
            ax.scatter(matchday, i, s=30, color=color)

    green_patch = mpatches.Patch(color=theme.GRASS_D, label="4 Points")
    orange_patch = mpatches.Patch(color=theme.GRASS, label="3 Points")
    blue_patch = mpatches.Patch(color=theme.GRASS_B, label="2 Points")
    gray_patch = mpatches.Patch(color="lightgray", label="0 Points")
    ax.legend(
        handles=[green_patch, orange_patch, blue_patch, gray_patch],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
