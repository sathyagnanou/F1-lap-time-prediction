"""
model_trainer.py
================
Training and evaluation of race-pace models.

  - Two baselines (season average, team average) that every model must beat.
  - Model selection with cross-validation grouped by race: every fold holds out
    whole races, so the model is always scored on tracks it has not seen.
  - A final evaluation on later races that were never used during selection.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import CATEGORICAL_FEATURES, TARGET

logger = logging.getLogger(__name__)


class TeamMeanBaseline(BaseEstimator, RegressorMixin):
    """Predicts the team's average pace seen in training (season average if unknown)."""

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.team_means_ = y.groupby(X["team_name"]).mean()
        self.global_mean_ = y.mean()
        return self

    def predict(self, X: pd.DataFrame):
        return X["team_name"].map(self.team_means_).fillna(self.global_mean_).to_numpy()


def _preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler() if scale else "passthrough", numeric),
    ])


def make_models(features: list[str]) -> dict:
    """Baselines and candidate models, each as a pipeline taking raw feature columns."""
    return {
        "Baseline: season average": DummyRegressor(strategy="mean"),
        "Baseline: team average": TeamMeanBaseline(),
        "Linear (Ridge)": make_pipeline(_preprocessor(features, scale=True), Ridge(alpha=1.0)),
        "Random Forest": make_pipeline(
            _preprocessor(features, scale=False),
            RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=42)),
        "Gradient Boosting": make_pipeline(
            _preprocessor(features, scale=False),
            GradientBoostingRegressor(random_state=42)),
    }


def regression_metrics(y_true, y_pred, pole_time_s) -> dict:
    """Errors in % of pole time, plus the MAE converted back to seconds."""
    return {
        "MAE_pct": mean_absolute_error(y_true, y_pred),
        "RMSE_pct": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAE_s": np.mean(np.abs(y_true - y_pred) / 100 * pole_time_s),
    }


class F1ModelTrainer:
    """
    Parameters
    ----------
    feature_sets : dict
        {name: list of feature columns}. Every model is evaluated on every set.
    n_folds : int
        Number of race-grouped cross-validation folds.
    """

    def __init__(self, feature_sets: dict[str, list[str]], n_folds: int = 5):
        self.feature_sets = feature_sets
        self.n_folds = n_folds
        self.cv_results: pd.DataFrame | None = None
        self.results: dict[str, dict] = {}

    def cross_validate(self, train: pd.DataFrame) -> pd.DataFrame:
        """Race-grouped CV on the training races, for every (feature set, model)."""
        rows = []
        folds = list(GroupKFold(n_splits=self.n_folds).split(train, groups=train["round"]))
        for set_name, features in self.feature_sets.items():
            for model_name, model in make_models(features).items():
                fold_mae, fold_train_mae = [], []
                for fit_idx, val_idx in folds:
                    fit, val = train.iloc[fit_idx], train.iloc[val_idx]
                    m = clone(model).fit(fit[features], fit[TARGET])
                    fold_mae.append(mean_absolute_error(val[TARGET], m.predict(val[features])))
                    fold_train_mae.append(mean_absolute_error(fit[TARGET], m.predict(fit[features])))
                rows.append({
                    "feature_set": set_name,
                    "model": model_name,
                    "cv_MAE_pct": np.mean(fold_mae),
                    "cv_MAE_std": np.std(fold_mae),
                    "train_MAE_pct": np.mean(fold_train_mae),
                })
                logger.info("CV %-28s %-26s MAE %.2f%%", set_name, model_name, np.mean(fold_mae))
        self.cv_results = pd.DataFrame(rows).sort_values("cv_MAE_pct").reset_index(drop=True)
        return self.cv_results

    def evaluate_on_test(self, train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> dict:
        """Fits every model on all training races and scores it once on the test races."""
        features = self.feature_sets[feature_set]
        self.results = {}
        for model_name, model in make_models(features).items():
            m = clone(model).fit(train[features], train[TARGET])
            y_pred = m.predict(test[features])
            self.results[model_name] = {
                "model": m,
                "features": features,
                "metrics": regression_metrics(test[TARGET], y_pred, test["pole_time_s"]),
                "y_true_test": test[TARGET].to_numpy(),
                "y_pred_test": y_pred,
            }
        return self.results

    def get_leaderboard(self) -> pd.DataFrame:
        rows = [{"Model": name, **res["metrics"]} for name, res in self.results.items()]
        return pd.DataFrame(rows).sort_values("MAE_pct").reset_index(drop=True)

    def get_feature_importance(self, model_name: str, test: pd.DataFrame) -> pd.DataFrame:
        """
        Permutation importance on the test races: how much the MAE grows when a
        feature is shuffled. Works for any model and is measured on unseen data.
        """
        res = self.results[model_name]
        imp = permutation_importance(
            res["model"], test[res["features"]], test[TARGET],
            scoring="neg_mean_absolute_error", n_repeats=10, random_state=42)
        return (pd.DataFrame({"feature": res["features"],
                              "importance": imp.importances_mean,
                              "std": imp.importances_std})
                .sort_values("importance", ascending=False).reset_index(drop=True))

    def error_by_group(self, model_name: str, test: pd.DataFrame, group: str) -> pd.DataFrame:
        """Mean absolute error and mean signed error (bias) per group, e.g. per race."""
        res = self.results[model_name]
        err = pd.DataFrame({group: test[group].to_numpy(),
                            "error": res["y_pred_test"] - res["y_true_test"]})
        return (err.groupby(group)["error"]
                .agg(MAE_pct=lambda e: e.abs().mean(), bias_pct="mean", laps="size")
                .sort_values("MAE_pct", ascending=False).reset_index())

    def save_models(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, res in self.results.items():
            fname = name.lower().replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
            joblib.dump(res["model"], path / f"{fname}.joblib")
        logger.info("Models saved to %s", path)
