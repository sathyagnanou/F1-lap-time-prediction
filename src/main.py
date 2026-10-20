"""
Runs the race-pace modelling pipeline end to end:

    python src/main.py

1. Load every 2024 race (OpenF1 + Ergast, cached in data/).
2. Clean the laps and build features.
3. Select a model with race-grouped cross-validation on rounds 1-18.
4. Evaluate it once on rounds 19-24.
5. Save tables to results/, figures to visualizations/ and models to models/.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from data_loader import load_season_race_laps
from feature_engineering import (FEATURE_SETS, TARGET, add_race_features, clean_race_laps,
                                 train_test_split_by_round)
from model_trainer import F1ModelTrainer
from visualizer import F1Visualizer

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
YEAR = 2024
FIRST_TEST_ROUND = 19


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s — %(message)s")
    for noisy in ["fastf1", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    RESULTS_DIR.mkdir(exist_ok=True)
    viz = F1Visualizer()

    # Data
    raw = load_season_race_laps(YEAR)
    laps, cleaning_report = clean_race_laps(raw)
    laps = add_race_features(laps)
    train, test = train_test_split_by_round(laps, FIRST_TEST_ROUND)
    cleaning_report.to_csv(RESULTS_DIR / "cleaning_report.csv", index=False)
    print(cleaning_report.to_string(index=False), "\n")

    viz.plot_target_by_race(laps, TARGET, FIRST_TEST_ROUND)
    viz.plot_pace_vs_tyre_age(train, TARGET)

    # Model selection on the training races only
    trainer = F1ModelTrainer(FEATURE_SETS)
    cv = trainer.cross_validate(train)
    cv.to_csv(RESULTS_DIR / "cross_validation.csv", index=False)
    viz.plot_cv_results(cv)
    print(cv.round(3).to_string(index=False), "\n")

    best = cv[~cv["model"].str.startswith("Baseline")].iloc[0]
    print(f"Selected: {best.model} with features '{best.feature_set}'\n")

    # Final evaluation on the test races
    trainer.evaluate_on_test(train, test, best.feature_set)
    leaderboard = trainer.get_leaderboard()
    leaderboard.to_csv(RESULTS_DIR / "test_leaderboard.csv", index=False)
    print(leaderboard.round(3).to_string(index=False), "\n")

    importance = trainer.get_feature_importance(best.model, test)
    by_race = trainer.error_by_group(best.model, test, "location")
    by_race.to_csv(RESULTS_DIR / "test_error_by_race.csv", index=False)
    print(by_race.round(2).to_string(index=False))

    viz.plot_predictions_vs_actual(trainer.results)
    viz.plot_residuals(trainer.results, best.model)
    viz.plot_feature_importances(importance, best.model)
    viz.plot_error_by_group(by_race, "location")
    trainer.save_models(ROOT_DIR / "models")


if __name__ == "__main__":
    main()
