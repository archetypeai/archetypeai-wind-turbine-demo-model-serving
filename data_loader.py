"""Load and sample Penmanshiel SCADA data.

The turbine CSVs have leading `# ...` comment lines, then a header row, then 10-minute
rows for the whole year. The status CSVs are a flat table of events.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Columns read from each turbine CSV for the replay. The classifier embeds four
# of them (see FEATURE_COLUMNS in newton_client.py).
SIGNAL_COLUMNS: list[str] = [
    "Date and time",
    "Wind speed (m/s)",
    "Power (kW)",
    "Rotor speed (RPM)",
    "Generator RPM (RPM)",
    "Gear oil temperature (°C)",
    "Generator bearing front temperature (°C)",
    "Generator bearing rear temperature (°C)",
    "Front bearing temperature (°C)",
    "Nacelle ambient temperature (°C)",
    "Grid frequency (Hz)",
    "Blade angle (pitch position) A (°)",
    "Capacity factor",
    "Nacelle position (°)",
    "Wind direction (°)",
]


@dataclass
class TurbineInfo:
    wt_id: str  # "01" .. "15"
    title: str  # "Penmanshiel 01"
    turbine_csv: str
    status_csv: str
    rated_power_kw: int


def _strip_comments_and_read(path: str) -> pd.DataFrame:
    """Read a Greenbyte-style CSV that has `# ...` comment lines before the header.

    The header line itself begins with `# ` (e.g. `# Date and time,Wind speed...`).
    We locate it as the *last* contiguous `#` line, strip the leading `# `, then read
    the rest via pandas.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            header_idx = i
        else:
            break
    header_line = lines[header_idx].lstrip("#").lstrip()
    data = [header_line] + lines[header_idx + 1:]
    from io import StringIO
    return pd.read_csv(StringIO("".join(data)), low_memory=False)


def discover_turbines() -> list[TurbineInfo]:
    """Enumerate Penmanshiel turbines from the SCADA folders + static metadata."""
    static = pd.read_csv(os.path.join(DATA_DIR, "Penmanshiel_WT_static.csv"))
    static = static.dropna(subset=["Title"])
    rated_by_title = {
        row["Title"]: int(row["Rated power (kW)"])
        for _, row in static.iterrows()
        if pd.notna(row.get("Rated power (kW)"))
    }

    turbines: list[TurbineInfo] = []
    for folder in ("Penmanshiel_SCADA_2019_WT01-10_3112", "Penmanshiel_SCADA_2019_WT11-15_3117"):
        for turbine_path in sorted(glob.glob(os.path.join(DATA_DIR, folder, "Turbine_Data_*.csv"))):
            fname = os.path.basename(turbine_path)
            # Turbine_Data_Penmanshiel_01_2019-01-01_..._1042.csv
            wt_id = fname.split("Penmanshiel_")[1].split("_")[0]
            title = f"Penmanshiel {wt_id}"
            status_path = turbine_path.replace("Turbine_Data_", "Status_")
            turbines.append(
                TurbineInfo(
                    wt_id=wt_id,
                    title=title,
                    turbine_csv=turbine_path,
                    status_csv=status_path if os.path.exists(status_path) else "",
                    rated_power_kw=rated_by_title.get(title, 2050),
                )
            )
    return turbines


def load_turbine_window(
    info: TurbineInfo,
    start: str,
    end: str,
    max_rows: int = 144,
) -> pd.DataFrame:
    """Return rows in [start, end), downsampled to ≤ max_rows. Times are ISO strings."""
    df = _strip_comments_and_read(info.turbine_csv)
    ts_col = "Date and time"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (df[ts_col] >= start_ts) & (df[ts_col] < end_ts)
    window = df.loc[mask, SIGNAL_COLUMNS].copy()

    if len(window) > max_rows:
        # Even-step downsample, rounding up so the full window is covered.
        step = (len(window) + max_rows - 1) // max_rows
        window = window.iloc[::step]
    return window.reset_index(drop=True)
