"""
Utility functions for football prediction models.
Extracted from 3_baseline.ipynb to avoid code duplication.
"""

import numpy as np


def kicktipp_scoring_rule(scoreline_actual, scoreline_pred):
    """Calculate Kicktipp points for a prediction."""
    home_goals, away_goals = scoreline_actual
    pred_home, pred_away = scoreline_pred
    if home_goals == away_goals:
        # draw
        if pred_home == home_goals and pred_away == away_goals:
            return 4
        elif (pred_home - pred_away) == 0:
            return 2
        else:
            return 0
    else:
        # not a draw
        if pred_home == home_goals and pred_away == away_goals:
            return 4
        elif (pred_home - pred_away) == (home_goals - away_goals):
            return 3
        elif (pred_home > pred_away) == (home_goals > away_goals):
            return 2
        else:
            return 0


def generate_kicktipp_score_matrix(scoreline, max_goals=4):
    """Generate score matrix for a specific predicted scoreline."""
    scores = []
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            scores.append(kicktipp_scoring_rule(scoreline, (hg, ag)))
    return np.array(scores).reshape((max_goals + 1, max_goals + 1))


def generate_kicktipp_score_matrices(max_goals=4):
    """Generate score matrices for all possible predictions."""
    matrices = []
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            matrices.append(
                generate_kicktipp_score_matrix((hg, ag), max_goals=max_goals)
            )
    return np.array(matrices).reshape(
        max_goals + 1, max_goals + 1, max_goals + 1, max_goals + 1
    )


# Cache for score matrices
kicktipp_score_matrices_for_max_goals = {}


def max_expected_reward_guesses(probabilities: np.ndarray, max_goals=4):
    """Find optimal predictions to maximize expected Kicktipp points."""
    # compute the score matrices only once and cache them for later use
    try:
        kicktipp_score_matrices = kicktipp_score_matrices_for_max_goals[max_goals]
    except KeyError:
        kicktipp_score_matrices = generate_kicktipp_score_matrices(max_goals=max_goals)
        kicktipp_score_matrices_for_max_goals[max_goals] = kicktipp_score_matrices

    expected_rewards = np.einsum(
        "uij,ijkl->ukl", probabilities, kicktipp_score_matrices
    )
    guesses = np.array(
        [
            np.unravel_index(
                np.argmax(expected_rewards[i]),
                expected_rewards.shape[1:],
            )
            for i in range(expected_rewards.shape[0])
        ]
    )
    return guesses


def compute_kicktipp_points(df_matches, predictions):
    """Compute total Kicktipp points for a set of predictions."""
    actual_scorelines = df_matches[["home_goals", "away_goals"]].to_numpy()
    points = [
        kicktipp_scoring_rule(actual, pred)
        for actual, pred in zip(actual_scorelines, predictions)
    ]
    return int(np.sum(points))


class BaselineModel:
    """Simple baseline model using historical scoreline frequencies."""

    def __init__(self, max_goals=4):
        self.max_goals = max_goals

    def fit(self, df_matches):
        scorelines = df_matches[["home_goals", "away_goals"]].to_numpy()
        scorelines = np.clip(scorelines, 0, self.max_goals)
        counts = np.zeros((self.max_goals + 1, self.max_goals + 1), dtype=int)
        for hg, ag in scorelines:
            counts[hg, ag] += 1
        self.counts_ = counts
        self.probabilities_ = counts / np.sum(counts)

    def predict_proba(self, df_matches):
        n = df_matches.shape[0]
        probabilities = np.broadcast_to(
            self.probabilities_, (n,) + self.probabilities_.shape
        )
        return probabilities

    def predict(self, df_matches):
        probabilities = self.predict_proba(df_matches)
        guesses = max_expected_reward_guesses(probabilities, max_goals=self.max_goals)
        return guesses


class CrossValidationResult:
    """Container for cross-validation results."""

    def __init__(self):
        self.scores = []

    @property
    def score_mean(self):
        return np.mean([score for year, score in self.scores]).item()

    @property
    def score_std(self):
        return np.std([score for year, score in self.scores]).item()

    def print_summary(self):
        print(f"Mean score: {self.score_mean:.1f} ± {self.score_std:.1f}")
        print(f"Individual scores: {[score for year, score in self.scores]}")


def cross_validate_model(model, df_matches, horizon=6):
    """Cross-validate a model across seasons."""
    years = df_matches["season"].unique()
    min_year, max_year = int(np.min(years)), int(np.max(years))
    result = CrossValidationResult()

    for year in range(min_year + 1, max_year + 1):
        df_train_cv = df_matches[
            (df_matches["season"] < year) & (df_matches["season"] >= year - horizon)
        ]
        df_test_cv = df_matches[df_matches["season"] == year]

        # Create a fresh model instance for each fold
        if hasattr(model, "max_goals"):
            model_cv = type(model)(max_goals=model.max_goals)
        else:
            model_cv = type(model)()

        model_cv.fit(df_train_cv)
        predictions = model_cv.predict(df_test_cv)
        score = compute_kicktipp_points(df_test_cv, predictions)
        result.scores.append((year, score))

    return result
