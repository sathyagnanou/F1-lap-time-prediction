"""
feature_engineering.py
======================
Turns raw data into analysis and model inputs.

  - Telemetry: summary features for a single lap (speed, throttle, braking...).
  - Race laps: cleaning rules and the features used to model race pace.

Modelling target
----------------
`pct_off_pole`: how much slower a race lap is than the fastest qualifying lap of
the same weekend, in %. Normalising by the pole time removes most of the
difference between tracks (a 70 s lap in Monaco vs a 105 s lap in Spa), and the
pole time is known *before* the race, so it does not leak race information.
Sector times are deliberately not used as features: they add up to the lap time.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MAX_PCT_OFF_POLE = 20.0  # slower laps are safety car / VSC / incident laps

TARGET = "pct_off_pole"
CATEGORICAL_FEATURES = ["compound", "team_name"]
RACE_STATE_FEATURES = ["tyre_age", "race_progress", "stint_number"]
WEATHER_FEATURES = ["track_temperature", "air_temperature", "humidity", "rainfall"]

FEATURE_SETS = {
    "tyres + race state": CATEGORICAL_FEATURES + RACE_STATE_FEATURES,
    "tyres + race state + weather": CATEGORICAL_FEATURES + RACE_STATE_FEATURES + WEATHER_FEATURES,
}


# ── Telemetry features ───────────────────────────────────────────────────
def extract_lap_features(tel: pd.DataFrame, full_throttle: float = 98.0,
                         corner_speed: float = 200.0) -> dict:
    """
    Summary features of one lap of telemetry (output of F1DataLoader.get_telemetry).

    full_throttle : throttle % above which the driver counts as flat out.
    corner_speed  : speed (km/h) below which the car counts as being in a corner.
    """
    dt = tel["Time"].diff().dt.total_seconds().fillna(0)
    braking = tel["Brake"].astype(bool)
    flat_out = tel["Throttle"] >= full_throttle
    in_corner = tel["Speed"] < corner_speed

    return {
        "Driver": tel["Driver"].iloc[0],
        "LapTime_s": tel["LapTime_s"].iloc[0],
        "speed_mean": tel["Speed"].mean(),
        "speed_max": tel["Speed"].max(),
        "speed_min": tel["Speed"].min(),
        "full_throttle_pct": flat_out.mean() * 100,
        "time_full_throttle_s": dt[flat_out].sum(),
        "brake_events": int((braking.astype(int).diff() > 0).sum()),
        "time_braking_s": dt[braking].sum(),
        "gear_changes": int(tel["nGear"].diff().abs().fillna(0).sum()),
        "time_in_corners_pct": in_corner.mean() * 100,
        "rpm_max": tel["RPM"].max(),
    }


# ── Race laps: cleaning ──────────────────────────────────────────────────
def clean_race_laps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keeps representative green-flag racing laps on dry tyres.

    Returns the cleaned laps and a report of how many laps each rule removed,
    so the effect of every cleaning decision is visible.
    """
    df = df.copy()
    df[TARGET] = (df["lap_duration"] / df["pole_time_s"] - 1) * 100

    rules = [
        ("missing lap time", df["lap_duration"].isna()),
        ("lap 1 (standing start)", df["lap_number"] == 1),
        ("pit out-lap", df["is_pit_out_lap"].astype(bool)),
        ("pit in-lap", df["is_pit_in_lap"].astype(bool)),
        ("wet qualifying (pole time not comparable)", df["quali_wet_share"] >= 0.5),
        ("intermediate / wet tyres", ~df["compound"].isin(DRY_COMPOUNDS)),
        (f"> {MAX_PCT_OFF_POLE:.0f}% off pole (safety car, incidents)",
         df[TARGET] > MAX_PCT_OFF_POLE),
    ]

    keep = pd.Series(True, index=df.index)
    report = []
    for name, drop in rules:
        removed = (keep & drop).sum()
        keep &= ~drop
        report.append({"rule": name, "laps_removed": int(removed), "laps_left": int(keep.sum())})

    report = pd.DataFrame(report)
    logger.info("Cleaning kept %d of %d laps", keep.sum(), len(df))
    return df[keep].reset_index(drop=True), report


# ── Race laps: features ──────────────────────────────────────────────────
def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds model features. Every feature is known at the start of the lap:
    tyre compound and age, team, how far into the race we are (a proxy for
    fuel load) and the current weather.
    """
    df = df.copy()
    df["race_progress"] = df["lap_number"] / df["race_laps"]
    df["rainfall"] = df["rainfall"].astype(float)
    return df


def train_test_split_by_round(df: pd.DataFrame, first_test_round: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: earlier races for training, later races for testing.
    A random split would put laps from the same race (same track, same day)
    in both sets and overstate how well the model generalises.
    """
    train = df[df["round"] < first_test_round].reset_index(drop=True)
    test = df[df["round"] >= first_test_round].reset_index(drop=True)
    return train, test
