import numpy as np
from sklearn.base import BaseEstimator

class StaticEstimator(BaseEstimator):
    """
    Always predict same result
    """
    def __init__(self, result) -> None:
        super().__init__()
        self.result = result

    def fit(self, X, y):
        pass

    def predict(self, X):
        results = np.zeros((len(X), 2))
        results[:] = self.result
        return results

class RandomEstimator(BaseEstimator):
    """
    Use the result distribution of the training data to predict new results.
    """
    def fit(self, X, y):
        df_result_counts = X[["host_goals", "guest_goals"]].value_counts().reset_index()
        df_result_counts["p"] = df_result_counts["count"] / len(X)
        self.df_result_counts = df_result_counts

    def predict(self, X):
        df_result_counts = self.df_result_counts
        result_indices = np.random.choice(df_result_counts.index, len(X), p=df_result_counts["p"].values)
        return df_result_counts.loc[result_indices, ["host_goals", "guest_goals"]].values

class LatestTableEstimator(BaseEstimator):
    """
    Predicts results depending on who got the highest place last season 
    """
    def fit(self, df_matches, y):
        last_season = df_matches["season"].max()
        table = df_matches.loc[df_matches["season"] == last_season, ["host_id", "host_last_season_goal_diff", "host_last_season_points"]].groupby("host_id").min().copy().reset_index()
        table = table.rename(columns={"host_last_season_points": "points", "host_last_season_goal_diff": "goal_diff", "host_id": "team_id"}).set_index("team_id")
        table = table.sort_values(by=["points", "goal_diff"], ascending=False)
        table["position"] = range(1, len(table) + 1)
        self.table = table
    
    def predict(self, X):
        results = np.zeros((len(X), 2))
        for i, (index, match) in enumerate(X.iterrows()):
            try:
                host_position = self.table.loc[match["host_id"], "position"]
            except KeyError as e:
                host_position = 20
            try:
                guest_position = self.table.loc[match["guest_id"], "position"]
            except KeyError as e:
                guest_position = 20
            
            if host_position <= guest_position:
                results[i] = [2, 1]
            else:
                results[i] = [1, 2]

        return results
            

