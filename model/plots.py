import numpy as np
import matplotlib.pyplot as plt

def cross_val_plot(df_matches, splits, labels, scores):
    scores = np.array(scores)
    sorting = np.argsort(-np.mean(scores, axis=1))
    labels = [labels[i] for i in sorting]
    scores = scores[sorting]

    for label, score in zip(labels, scores):
        print(f"{label}:\t{score.mean():.3f} +- {score.std():.3f}")

    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    split_test_seasons = [df_matches.iloc[split[1][0]]["season"] for  split in splits]

    axs[0].plot(scores.T, label=labels)
    axs[0].set_xticks(range(len(split_test_seasons)), split_test_seasons, rotation=90, ha='center')
    axs[0].set_xlabel("Test season")
    axs[0].set_ylabel("Score")
    axs[0].legend()

    axs[1].boxplot(scores.T)
    axs[1].set_xlabel("Strategy")
    axs[1].set_xticks(range(1, len(labels) +1), labels, rotation=45, ha="center")
    fig.tight_layout()
    return fig, axs