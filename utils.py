import numpy as np


def cross_val_splits(df, column="leagueSeason", start=1):
    df = df.reset_index()
    splits = []
    split_test_seasons = []
    values_unique = df[column].unique()
    values_unique.sort()
    min_value = values_unique[0]
    for i in range(start, len(values_unique)):
        test_value = values_unique[i]
        train_rows = df[(df[column] >= min_value) & (df[column] < test_value)]
        test_rows = df[df[column] == test_value]
        splits.append((train_rows.index, test_rows.index))
        split_test_seasons.append(values_unique[i])
    return splits

def compute_kicktipp_score(predictions, y):
    predictions_draw = predictions[:, 0] == predictions[:, 1]
    predictions_win1 = predictions[:, 0] > predictions[:, 1]
    predictions_win2 = predictions[:, 0] < predictions[:, 1]
    predictions_diff = predictions[:, 0] - predictions[:, 1]
    y_draw = y[:, 0] == y[:, 1]
    y_win1 = y[:, 0] > y[:, 1]
    y_win2 = y[:, 0] < y[:, 1]
    y_diff = y[:, 0] - y[:, 1]

    correct_result = (predictions[:, 0] == y[:, 0]) & (predictions[:, 1] == y[:, 1])
    correct_result_win = (correct_result & y_win1) | (correct_result & y_win2)
    correct_result_draw = correct_result & (~correct_result_win)
    correct_tendency_win = (predictions_win1 & y_win1) | (predictions_win2 & y_win2)
    correct_tendency_draw = (predictions_draw & y_draw)
    correct_diff_win = (predictions_diff == y_diff) & correct_tendency_win

    score = 0
    score += np.count_nonzero(correct_result) * 4
    score += np.count_nonzero(correct_diff_win & (~correct_result)) * 3
    score += np.count_nonzero(correct_tendency_draw & (~correct_result_draw)) * 2
    score += np.count_nonzero(correct_tendency_win & (~correct_result_win) & ~(correct_diff_win)) * 2
    return score

def kicktipp_scoring(estimator, X, y):
    predictions = estimator.predict(X)
    return compute_kicktipp_score(predictions, y)

assert compute_kicktipp_score(np.array([[0, 0], [1, 1]]), np.array([[0, 0], [1, 1]])) == 8
assert compute_kicktipp_score(np.array([[0, 0], [2, 1]]), np.array([[0, 0], [1, 1]])) == 4
assert compute_kicktipp_score(np.array([[0, 0], [1, 1]]), np.array([[0, 0], [1, 2]])) == 4
assert compute_kicktipp_score(np.array([[0, 0], [2, 1]]), np.array([[0, 0], [1, 0]])) == 7
assert compute_kicktipp_score(np.array([[1, 0], [2, 1]]), np.array([[0, 0], [1, 0]])) == 3
assert compute_kicktipp_score(np.array([[1, 3], [1, 1], [3, 0], [1, 1], [0, 2], [1, 1], [3, 1], [1, 3], [1, 1]]), np.array([[2, 3], [3, 2], [1, 0], [2, 2], [3, 1], [1, 1], [2, 0], [2, 3], [0, 2]])) == 15