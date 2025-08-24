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
        for actual, pred in zip(actual_scorelines, predictions, strict=False)
    ]
    return int(np.sum(points))


def to_predictions_df(df_matches, predictions):
    df_predictions = df_matches.copy()
    df_predictions["home_goals_pred"] = predictions[:, 0]
    df_predictions["away_goals_pred"] = predictions[:, 1]
    predictions = df_predictions[["home_goals_pred", "away_goals_pred"]].values
    labels = df_predictions[["home_goals", "away_goals"]].values
    kicktipp_scores = np.zeros(len(labels))
    for i in range(len(predictions)):
        pred = (predictions[i, 0], predictions[i, 1])
        ground_truth = (labels[i, 0], labels[i, 1])
        kicktipp_scores[i] = kicktipp_scoring_rule(ground_truth, pred)
    df_predictions["score"] = kicktipp_scores
    return df_predictions
