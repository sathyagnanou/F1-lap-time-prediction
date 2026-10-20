"""
visualizer.py
=============
Plots for the telemetry analysis and the race-pace model.

Telemetry (one qualifying session):
  1. Circuit coloured by speed
  2. Braking and throttle zones on the circuit
  3. Telemetry comparison between two drivers
  4. Speed trace of several drivers
  5. Sector analysis

Race-pace model (whole season):
  6. Race pace by track
  7. Pace vs tyre age per compound
  8. Cross-validation results (overfitting check)
  9. Predictions vs actual
 10. Residuals
 11. Feature importances
 12. Error by race
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

logger = logging.getLogger(__name__)

VIZ_DIR = Path(__file__).resolve().parents[1] / "visualizations"

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#444444",
    "axes.labelcolor":  "#eeeeee",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#aaaaaa",
    "ytick.color":      "#aaaaaa",
    "text.color":       "#eeeeee",
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#444444",
    "grid.color":       "#2a2a2a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "monospace",
    "figure.dpi":       110,
})

DRIVER_COLORS = ["#FF8000", "#00D2FF", "#E8002D", "#27F4D2", "#FFD700", "#BB86FC"]
COMPOUND_COLORS = {"SOFT": "#E8002D", "MEDIUM": "#FFD700", "HARD": "#EEEEEE"}
BASELINE_COLOR = "#777777"
MODEL_COLOR = "#FF8000"


class F1Visualizer:
    """
    Parameters
    ----------
    save_dir : Path, optional
        Where figures are saved. Default: visualizations/.
    dpi : int
        Resolution of saved figures.
    """

    def __init__(self, save_dir: Optional[Path] = None, dpi: int = 150):
        self.save_dir = Path(save_dir or VIZ_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def _save(self, fig: plt.Figure, name: str, save: bool):
        if save:
            path = self.save_dir / f"{name}.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved %s", path)

    @staticmethod
    def _track_axes(ax, x, y):
        ax.set_xlim(x.min() - 100, x.max() + 100)
        ax.set_ylim(y.min() - 100, y.max() + 100)
        ax.set_aspect("equal")
        ax.axis("off")

    @staticmethod
    def _colored_line(x, y, values, cmap, linewidth=3):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(values.min(), values.max()),
                            linewidth=linewidth)
        lc.set_array(values)
        return lc

    # ── 1. Circuit coloured by speed ──────────────────────────────────────
    def plot_circuit_speed(self, tel: pd.DataFrame, driver: str,
                           title: Optional[str] = None, save: bool = True) -> plt.Figure:
        """Requires columns X, Y, Speed."""
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_facecolor("#0f0f0f")
        x, y = tel["X"].values, tel["Y"].values
        lc = self._colored_line(x, y, tel["Speed"].values, "plasma")
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="Speed (km/h)", fraction=0.03, pad=0.02)
        self._track_axes(ax, x, y)
        ax.set_title(title or f"Speed on fastest lap — {driver}", fontsize=13, fontweight="bold")
        self._save(fig, f"circuit_speed_{driver}", save)
        return fig

    # ── 2. Braking and throttle zones ─────────────────────────────────────
    def plot_braking_acceleration_zones(self, tel: pd.DataFrame, driver: str,
                                        save: bool = True) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        x, y = tel["X"].values, tel["Y"].values
        panels = [
            (tel["Brake"].astype(float).values, "Reds", "Brake (on / off)"),
            (tel["Throttle"].values, "RdYlGn", "Throttle (%)"),
        ]
        for ax, (values, cmap, label) in zip(axes, panels):
            ax.set_facecolor("#0f0f0f")
            ax.plot(x, y, color="#333333", linewidth=6, zorder=1)
            lc = self._colored_line(x, y, values, cmap, linewidth=4)
            ax.add_collection(lc)
            fig.colorbar(lc, ax=ax, label=label, fraction=0.03)
            self._track_axes(ax, x, y)
            ax.set_title(label, fontsize=12)
        fig.suptitle(f"Braking and throttle zones — {driver}", fontsize=14)
        plt.tight_layout()
        self._save(fig, f"braking_throttle_{driver}", save)
        return fig

    # ── 3. Telemetry comparison ───────────────────────────────────────────
    def plot_telemetry_comparison(self, tel1: pd.DataFrame, tel2: pd.DataFrame,
                                  driver1: str, driver2: str, save: bool = True) -> plt.Figure:
        """Speed, throttle, brake, RPM and gear against distance for two drivers."""
        channels = [("Speed", "Speed (km/h)"), ("Throttle", "Throttle (%)"),
                    ("Brake", "Brake"), ("RPM", "RPM"), ("nGear", "Gear")]
        fig, axes = plt.subplots(len(channels), 1, figsize=(14, 13), sharex=True)
        for ax, (col, ylabel) in zip(axes, channels):
            ax.plot(tel1["Distance"], tel1[col].astype(float), color=DRIVER_COLORS[0],
                    linewidth=1.4, label=driver1)
            ax.plot(tel2["Distance"], tel2[col].astype(float), color=DRIVER_COLORS[1],
                    linewidth=1.4, label=driver2)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True)
        axes[0].legend(loc="lower right")
        axes[-1].set_xlabel("Distance (m)")
        fig.suptitle(f"Fastest lap telemetry: {driver1} vs {driver2}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, f"telemetry_{driver1}_vs_{driver2}", save)
        return fig

    # ── 4. Speed trace ────────────────────────────────────────────────────
    def plot_speed_trace(self, telemetry_dict: dict[str, pd.DataFrame],
                         save: bool = True) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 5))
        for i, (driver, tel) in enumerate(telemetry_dict.items()):
            ax.plot(tel["Distance"], tel["Speed"], color=DRIVER_COLORS[i % len(DRIVER_COLORS)],
                    linewidth=1.6, label=driver)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title("Speed trace — fastest lap")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        self._save(fig, "speed_trace_comparison", save)
        return fig

    # ── 5. Sector analysis ────────────────────────────────────────────────
    def plot_sector_analysis(self, sector_df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """Gap to the best time in each sector, per driver (from F1DataLoader.get_sector_times)."""
        df = sector_df.copy()
        gap_cols = ["Sector1_gap", "Sector2_gap", "Sector3_gap"]
        df["total_gap"] = df[gap_cols].sum(axis=1)
        df = df.sort_values("total_gap")

        fig, ax = plt.subplots(figsize=(7, max(5, len(df) * 0.35)))
        data = df[gap_cols].values
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0,
                       vmax=np.percentile(data, 90))
        ax.set_yticks(range(len(df)), df["Driver"])
        ax.set_xticks(range(3), ["Sector 1", "Sector 2", "Sector 3"])
        for r in range(len(df)):
            for c in range(3):
                ax.text(c, r, f"+{data[r, c]:.3f}", ha="center", va="center", fontsize=8,
                        color="black")
        fig.colorbar(im, ax=ax, label="Gap to best sector (s)")
        ax.set_title("Gap to the best time in each sector")
        plt.tight_layout()
        self._save(fig, "sector_analysis", save)
        return fig

    # ── 6. Race pace by track ─────────────────────────────────────────────
    def plot_target_by_race(self, laps: pd.DataFrame, target: str,
                            first_test_round: int, save: bool = True) -> plt.Figure:
        """Distribution of race pace (% off pole) for every race, train vs test."""
        order = laps.groupby("round")["location"].first()
        data = [laps.loc[laps["round"] == r, target].values for r in order.index]

        fig, ax = plt.subplots(figsize=(14, 5))
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for box, rnd in zip(bp["boxes"], order.index):
            box.set_facecolor(MODEL_COLOR if rnd >= first_test_round else "#3671C6")
            box.set_alpha(0.8)
        for element in ["whiskers", "caps", "medians"]:
            plt.setp(bp[element], color="#cccccc")
        ax.set_xticks(range(1, len(order) + 1), order.values, rotation=45, ha="right")
        ax.set_ylabel("Race lap time, % slower than pole")
        ax.set_title("Race pace by track  (blue = training races, orange = test races)")
        ax.grid(axis="y")
        plt.tight_layout()
        self._save(fig, "race_pace_by_track", save)
        return fig

    # ── 7. Pace vs tyre age ───────────────────────────────────────────────
    def plot_pace_vs_tyre_age(self, laps: pd.DataFrame, target: str,
                              save: bool = True) -> plt.Figure:
        """Average pace by tyre age for each dry compound (ages with ≥ 30 laps)."""
        fig, ax = plt.subplots(figsize=(11, 5))
        for compound, color in COMPOUND_COLORS.items():
            sub = laps[laps["compound"] == compound]
            stats = sub.groupby("tyre_age")[target].agg(["mean", "size"])
            stats = stats[stats["size"] >= 30]
            ax.plot(stats.index, stats["mean"], color=color, marker="o", markersize=3,
                    linewidth=1.5, label=f"{compound} ({len(sub):,} laps)")
        ax.set_xlabel("Tyre age (laps)")
        ax.set_ylabel("Mean % slower than pole")
        ax.set_title("Pace vs tyre age per compound — training races")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        self._save(fig, "pace_vs_tyre_age", save)
        return fig

    # ── 8. Cross-validation results ───────────────────────────────────────
    def plot_cv_results(self, cv_results: pd.DataFrame, save: bool = True) -> plt.Figure:
        """Train vs held-out-race error for every model and feature set."""
        sets = list(cv_results["feature_set"].unique())
        fig, axes = plt.subplots(1, len(sets), figsize=(7 * len(sets), 5), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, set_name in zip(axes, sets):
            df = cv_results[cv_results["feature_set"] == set_name].sort_values("cv_MAE_pct")
            y = np.arange(len(df))
            ax.barh(y + 0.2, df["train_MAE_pct"], 0.4, color="#3671C6", label="training races")
            ax.barh(y - 0.2, df["cv_MAE_pct"], 0.4, xerr=df["cv_MAE_std"], color=MODEL_COLOR,
                    ecolor="#cccccc", label="held-out races (CV)")
            ax.set_yticks(y, df["model"])
            ax.invert_yaxis()
            ax.set_xlabel("MAE (% of pole time)")
            ax.set_title(f"Features: {set_name}")
            ax.grid(axis="x")
        fig.suptitle("Race-grouped cross-validation — a big gap between bars = overfitting",
                     fontsize=13)
        plt.tight_layout()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
        self._save(fig, "cross_validation", save)
        return fig

    # ── 9. Predictions vs actual ──────────────────────────────────────────
    def plot_predictions_vs_actual(self, results: dict, save: bool = True) -> plt.Figure:
        """Scatter of true vs predicted values for every model (F1ModelTrainer.results)."""
        n = len(results)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.array(axes).flatten()

        lims = [0, 20]
        for ax, (name, res) in zip(axes, results.items()):
            m = res["metrics"]
            ax.scatter(res["y_true_test"], res["y_pred_test"], alpha=0.25, s=6,
                       color=MODEL_COLOR, edgecolors="none")
            ax.plot(lims, lims, "w--", linewidth=1, label="perfect prediction")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Actual (% off pole)")
            ax.set_ylabel("Predicted (% off pole)")
            ax.set_title(f"{name}\nMAE {m['MAE_pct']:.2f}% ({m['MAE_s']:.2f}s)  R² {m['R2']:.2f}",
                         fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle("Predictions vs actual — test races", fontsize=13)
        plt.tight_layout()
        self._save(fig, "predictions_vs_actual", save)
        return fig

    # ── 10. Residuals ─────────────────────────────────────────────────────
    def plot_residuals(self, results: dict, model_name: str, save: bool = True) -> plt.Figure:
        res = results[model_name]
        residual = res["y_true_test"] - res["y_pred_test"]

        fig, (ax_hist, ax_sc) = plt.subplots(1, 2, figsize=(14, 5))
        ax_hist.hist(residual, bins=60, color="#3671C6", edgecolor="#111")
        ax_hist.axvline(0, color="red", linestyle="--")
        ax_hist.axvline(residual.mean(), color=MODEL_COLOR,
                        label=f"mean = {residual.mean():+.2f}")
        ax_hist.set_xlabel("Residual = actual − predicted (% of pole)")
        ax_hist.set_ylabel("Laps")
        ax_hist.legend()
        ax_hist.grid(axis="y")

        ax_sc.scatter(res["y_pred_test"], residual, alpha=0.25, s=6, color=MODEL_COLOR,
                      edgecolors="none")
        ax_sc.axhline(0, color="red", linestyle="--")
        ax_sc.set_xlabel("Predicted (% of pole)")
        ax_sc.set_ylabel("Residual")
        ax_sc.grid(True)
        fig.suptitle(f"Residuals — {model_name} — test races", fontsize=13)
        plt.tight_layout()
        self._save(fig, "residuals", save)
        return fig

    # ── 11. Feature importances ───────────────────────────────────────────
    def plot_feature_importances(self, importances_df: pd.DataFrame, model_name: str,
                                 save: bool = True) -> plt.Figure:
        """Bar chart of permutation importances (F1ModelTrainer.get_feature_importance)."""
        df = importances_df.iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.5)))
        ax.barh(df["feature"], df["importance"], xerr=df["std"], color=MODEL_COLOR,
                ecolor="#cccccc")
        ax.set_xlabel("Increase in MAE when the feature is shuffled (% of pole)")
        ax.set_title(f"Permutation feature importance — {model_name}")
        ax.grid(axis="x")
        plt.tight_layout()
        self._save(fig, "feature_importance", save)
        return fig

    # ── 12. Error by race ─────────────────────────────────────────────────
    def plot_error_by_group(self, error_df: pd.DataFrame, group: str,
                            save: bool = True) -> plt.Figure:
        """MAE and bias per group (F1ModelTrainer.error_by_group)."""
        df = error_df.iloc[::-1]
        y = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.6)))
        ax.barh(y + 0.2, df["MAE_pct"], 0.4, color=MODEL_COLOR, label="MAE")
        ax.barh(y - 0.2, df["bias_pct"], 0.4, color="#3671C6",
                label="bias (> 0 = predicted too slow)")
        ax.axvline(0, color="#cccccc", linewidth=1)
        ax.set_yticks(y, df[group])
        ax.set_xlabel("% of pole time")
        ax.set_title(f"Test error by {group}")
        ax.legend()
        ax.grid(axis="x")
        plt.tight_layout()
        self._save(fig, f"error_by_{group}", save)
        return fig
