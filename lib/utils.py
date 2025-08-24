import numpy as np
import requests
from tqdm import trange

from lib.scores import compute_kicktipp_points


def download_from_openligadb(
    year_from: int, year_to: int, leagues: list[str]
) -> list[dict]:
    """Download match data from OpenLigaDB for the specified years and leagues."""
    matches = []
    for season in trange(year_from, year_to + 1, desc="Downloading matches"):
        for league in leagues:
            response = requests.get(
                f"https://api.openligadb.de/getmatchdata/{league}/{season}"
            )
            matches += response.json()
    return matches


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
