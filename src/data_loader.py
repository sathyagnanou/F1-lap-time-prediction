"""
data_loader.py
==============
Data loading for the project.

Two sources are used:
  - FastF1: detailed car telemetry (speed, throttle, brake, GPS) for one
    qualifying session, used for the exploratory telemetry analysis.
  - OpenF1 (https://openf1.org): lap times, tyre stints, teams and weather for
    every race of a season, used to build the modelling dataset.

Raw API responses are cached on disk so the pipeline only hits the network once.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import fastf1
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
FASTF1_CACHE_DIR = ROOT_DIR / "data" / "cache"
OPENF1_CACHE_DIR = ROOT_DIR / "data" / "raw" / "openf1"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ERGAST_CACHE_DIR = ROOT_DIR / "data" / "raw" / "ergast"

OPENF1_URL = "https://api.openf1.org/v1"
ERGAST_URL = "https://api.jolpi.ca/ergast/f1"


# ── FastF1: telemetry for a single session ────────────────────────────────
class F1DataLoader:
    """
    Loads one session with FastF1 and exposes laps and telemetry.

    Parameters
    ----------
    year : int
        Season (e.g. 2024).
    grand_prix : str
        Grand Prix name (e.g. 'Monaco').
    session_type : str
        'FP1', 'FP2', 'FP3', 'Q', 'S' or 'R'.
    """

    def __init__(self, year: int, grand_prix: str, session_type: str = "Q"):
        self.year = year
        self.grand_prix = grand_prix
        self.session_type = session_type
        self.session = None

    def load_session(self) -> "F1DataLoader":
        FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
        logger.info("Loading %s %s %s", self.year, self.grand_prix, self.session_type)
        self.session = fastf1.get_session(self.year, self.grand_prix, self.session_type)
        self.session.load(telemetry=True, weather=True, messages=False, laps=True)
        return self

    def _require_session(self):
        if self.session is None:
            raise RuntimeError("Call load_session() first.")

    def get_laps(self, driver: Optional[str] = None) -> pd.DataFrame:
        """Quick laps (within 107% of the fastest lap), optionally for one driver."""
        self._require_session()
        laps = self.session.laps
        if driver:
            laps = laps.pick_drivers(driver)
        return laps.pick_quicklaps(threshold=1.07).reset_index(drop=True)

    def get_telemetry(self, driver: str, lap_number: Optional[int] = None) -> pd.DataFrame:
        """
        Telemetry of a driver's fastest lap (or of a given lap), with distance
        and GPS position merged in.
        """
        self._require_session()
        laps = self.session.laps.pick_drivers(driver)
        if lap_number is not None:
            lap = laps[laps["LapNumber"] == lap_number].iloc[0]
        else:
            lap = laps.pick_fastest()

        tel = lap.get_telemetry()
        tel["Driver"] = driver
        tel["LapNumber"] = lap["LapNumber"]
        tel["LapTime_s"] = lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else np.nan
        return tel

    def get_drivers_summary(self) -> pd.DataFrame:
        """Best lap, team, number of quick laps and gap to the fastest driver."""
        self._require_session()
        records = []
        for drv in self.session.laps["Driver"].unique():
            drv_laps = self.get_laps(drv)
            if drv_laps.empty:
                continue
            records.append({
                "Driver": drv,
                "Team": drv_laps["Team"].iloc[0],
                "BestLapTime_s": drv_laps["LapTime"].min().total_seconds(),
                "LapCount": len(drv_laps),
            })
        df = pd.DataFrame(records).sort_values("BestLapTime_s").reset_index(drop=True)
        df["Gap_s"] = df["BestLapTime_s"] - df["BestLapTime_s"].iloc[0]
        return df

    def get_sector_times(self) -> pd.DataFrame:
        """Sector times in seconds of each driver's fastest lap."""
        self._require_session()
        rows = []
        for drv in self.session.laps["Driver"].unique():
            best = self.session.laps.pick_drivers(drv).pick_fastest()
            if best is None or pd.isna(best["LapTime"]):
                continue
            rows.append({
                "Driver": drv,
                "Sector1_best": best["Sector1Time"].total_seconds(),
                "Sector2_best": best["Sector2Time"].total_seconds(),
                "Sector3_best": best["Sector3Time"].total_seconds(),
            })
        df = pd.DataFrame(rows).dropna()
        for sec in ["Sector1", "Sector2", "Sector3"]:
            df[f"{sec}_gap"] = df[f"{sec}_best"] - df[f"{sec}_best"].min()
        return df.sort_values("Sector1_best").reset_index(drop=True)


def load_telemetry_sample(year: int, grand_prix: str, session_type: str,
                          drivers: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Fastest-lap telemetry of a few drivers plus every driver's sector times.

    Saved to data/processed the first time, so the analysis can be re-run
    without the FastF1 cache (F1's live-timing server now rejects new downloads).
    """
    name = f"{year}_{grand_prix}_{session_type}".lower()
    tel_file = PROCESSED_DIR / f"telemetry_{name}.csv"
    sectors_file = PROCESSED_DIR / f"sectors_{name}.csv"

    if not (tel_file.exists() and sectors_file.exists()):
        loader = F1DataLoader(year, grand_prix, session_type).load_session()
        cols = ["Driver", "LapNumber", "LapTime_s", "Time", "Distance", "Speed", "RPM",
                "nGear", "Throttle", "Brake", "DRS", "X", "Y"]
        tel = pd.concat([loader.get_telemetry(d)[cols] for d in drivers])
        tel["Time"] = tel["Time"].dt.total_seconds()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        tel.to_csv(tel_file, index=False)
        loader.get_sector_times().to_csv(sectors_file, index=False)

    tel = pd.read_csv(tel_file)
    tel["Time"] = pd.to_timedelta(tel["Time"], unit="s")
    telemetry = {d: tel[tel["Driver"] == d].reset_index(drop=True) for d in drivers}
    return telemetry, pd.read_csv(sectors_file)


# ── OpenF1: season-wide race laps ─────────────────────────────────────────
def _cached_get(url: str, cache_file: Path, params: Optional[dict] = None):
    """GET a JSON API, with an on-disk cache and retry on rate limits."""
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    for attempt in range(7):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        data = resp.json()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
        time.sleep(1.0)  # stay under the free-tier rate limits
        return data
    raise RuntimeError(f"Rate limit: gave up on {url} {params}")


def _openf1_get(endpoint: str, **params) -> list[dict]:
    key = "_".join([endpoint] + [f"{k}-{v}" for k, v in sorted(params.items())])
    return _cached_get(f"{OPENF1_URL}/{endpoint}", OPENF1_CACHE_DIR / f"{key}.json", params)


def _to_seconds(lap_time: str) -> float:
    """'1:12.000' -> 72.0"""
    minutes, seconds = lap_time.split(":")
    return int(minutes) * 60 + float(seconds)


def _fastest_qualifying_lap(year: int, rnd: int) -> float:
    """
    Fastest valid qualifying lap (any of Q1/Q2/Q3) from the official results.
    OpenF1 lap times include deleted laps, so the Ergast results are used instead.
    """
    data = _cached_get(f"{ERGAST_URL}/{year}/{rnd}/qualifying.json",
                       ERGAST_CACHE_DIR / f"qualifying_{year}_{rnd}.json")
    results = data["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"]
    times = [_to_seconds(r[q]) for r in results for q in ("Q1", "Q2", "Q3") if r.get(q)]
    return min(times)


def _race_laps(race: dict, pole_time_s: float) -> pd.DataFrame:
    """All laps of one race, joined with tyre, team and weather information."""
    key = race["session_key"]
    laps = pd.DataFrame(_openf1_get("laps", session_key=key))
    stints = pd.DataFrame(_openf1_get("stints", session_key=key))
    drivers = pd.DataFrame(_openf1_get("drivers", session_key=key))
    weather = pd.DataFrame(_openf1_get("weather", session_key=key))

    laps = laps[["driver_number", "lap_number", "date_start", "lap_duration", "is_pit_out_lap"]]

    # Tyre compound and age: each lap is matched to the stint that contains it
    stints = stints.dropna(subset=["lap_start", "lap_end"])
    merged = laps.merge(stints[["driver_number", "stint_number", "lap_start", "lap_end",
                                "compound", "tyre_age_at_start"]], on="driver_number")
    merged = merged[(merged["lap_number"] >= merged["lap_start"])
                    & (merged["lap_number"] <= merged["lap_end"])]
    merged["tyre_age"] = merged["tyre_age_at_start"] + merged["lap_number"] - merged["lap_start"]

    # In-lap = last lap of a stint that is followed by another stint (driver pits)
    last_stint = stints.groupby("driver_number")["stint_number"].max().rename("last_stint")
    merged = merged.join(last_stint, on="driver_number")
    merged["is_pit_in_lap"] = (merged["lap_number"] == merged["lap_end"]) & (
        merged["stint_number"] < merged["last_stint"])

    merged = merged.merge(drivers[["driver_number", "name_acronym", "team_name"]],
                          on="driver_number", how="left")

    # Weather: latest reading before the start of the lap
    merged["date_start"] = pd.to_datetime(merged["date_start"], format="ISO8601", utc=True)
    weather["date"] = pd.to_datetime(weather["date"], format="ISO8601", utc=True)
    merged = merged.dropna(subset=["date_start"]).sort_values("date_start")
    merged = pd.merge_asof(
        merged, weather[["date", "track_temperature", "air_temperature", "humidity", "rainfall"]]
        .sort_values("date"),
        left_on="date_start", right_on="date", direction="backward",
    )

    merged["round"] = race["round"]
    merged["location"] = race["location"]
    merged["race_laps"] = laps["lap_number"].max()
    merged["pole_time_s"] = pole_time_s
    return merged.drop(columns=["lap_start", "lap_end", "tyre_age_at_start", "last_stint", "date"])


def load_season_race_laps(year: int = 2024, force: bool = False) -> pd.DataFrame:
    """
    Builds (or reads from data/processed) one row per lap for every race of the season.

    Columns: round, location, driver, team, lap number, lap time, tyre compound
    and age, weather, and the pole time from that weekend's qualifying.
    Laps are not cleaned here — see feature_engineering.clean_race_laps().
    """
    out_file = PROCESSED_DIR / f"race_laps_{year}.csv"
    if out_file.exists() and not force:
        return pd.read_csv(out_file, parse_dates=["date_start"])

    races = sorted(_openf1_get("sessions", year=year, session_name="Race"),
                   key=lambda s: s["date_start"])
    qualis = {q["meeting_key"]: q for q in
              _openf1_get("sessions", year=year, session_name="Qualifying")}

    frames = []
    for rnd, race in enumerate(races, start=1):
        race["round"] = rnd
        quali = qualis[race["meeting_key"]]
        quali_stints = pd.DataFrame(_openf1_get("stints", session_key=quali["session_key"]))
        pole_time_s = _fastest_qualifying_lap(year, rnd)
        logger.info("Round %2d %-18s pole %.3fs", rnd, race["location"], pole_time_s)
        race_df = _race_laps(race, pole_time_s)
        # A wet qualifying makes the pole time a poor reference for the race pace
        race_df["quali_wet_share"] = quali_stints["compound"].isin(["INTERMEDIATE", "WET"]).mean()
        frames.append(race_df)

    df = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    logger.info("Saved %d laps to %s", len(df), out_file)
    return df
