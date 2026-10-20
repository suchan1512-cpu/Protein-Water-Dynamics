#!/usr/bin/env python3
"""
HDX HNH processing, normalization QC, scaling diagnostics, and robust fitting.

What this script checks
-----------------------
1. WT exchanged peaks are divided by the WT unexchanged/main HNH peaks.
2. MUT exchanged peaks are divided by the MUT unexchanged/main HNH peaks.
3. The acquisition correction factors are applied exactly as configured.
4. Missing/non-positive main-reference peaks and duplicate assignments are reported.
5. The percent-intensity calculation is independently recomputed and verified.
6. S/N < threshold points are flagged as below detection, not silently deleted.
   By default they are excluded from the kinetic fits (EXCLUDE_LOW_SNR_FROM_FIT);
   use --include-low-snr to fit all positive points regardless of S/N.
7. WT-vs-MUT multiplicative and additive global offsets are assessed.
8. Late-time positive floors are summarized.
9. Three robust kinetic models are compared:
      free plateau:       plateau + amplitude * exp(-k*t)
      zero plateau:       amplitude * exp(-k*t)
      fixed 100 -> zero:  100 * exp(-k*t)
10. Clean per-residue plots use zero-plateau fits by default and show a positive
    plateau only when strongly supported by AICc.
11. Per-residue and global QC plots, CSV tables, a text report, and a ZIP are written.

Important
---------
This script diagnoses a systematic offset. It does not automatically subtract a
baseline or force every curve to zero. A correction should only be applied after
the acquisition/processing cause is identified.
"""

from __future__ import annotations

import math
import re
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr, theilslopes


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_PATH = Path("examples/synthetic_hdx")
OUTPUT_DIR = Path("output/hdx_kinetics_qc")

INTENSITY_COLUMN = "Height"
MAX_TIME_FOR_PROCESSING = 76.0
PLOT_X_AXIS_MAX = 80.0

# Acquisition scaling used in the original pipeline
MAIN_INTENSITY_DIVISOR = 4.0

# Points below this S/N are flagged as below detection.
# They are shown in plots but excluded from the default kinetic fits.
SNR_THRESHOLD = 3.0

# If False, low-S/N points are kept in the fits and only flagged for QC.
EXCLUDE_LOW_SNR_FROM_FIT = True

# Late-time floor is assessed from this time onward.
# Set to None to use the upper quartile of available time points.
LATE_TIME_START: Optional[float] = 48.0

# Minimum number of usable observations needed for fitting
MIN_POINTS_FOR_FIT = 4

# Robust least-squares settings
ROBUST_LOSS = "soft_l1"
ROBUST_F_SCALE = 5.0
MAX_K = 10.0
MAX_AMPLITUDE = 300.0
MAX_PLATEAU = 200.0

# Heuristic QC thresholds. These only create flags; they do not alter data.
GLOBAL_RATIO_DEVIATION_THRESHOLD = 0.15       # 15% away from ratio 1
GLOBAL_LOG_RATIO_MAD_THRESHOLD = 0.15         # low spread across residues
ADDITIVE_OFFSET_THRESHOLD = 10.0              # percentage points
LATE_FLOOR_THRESHOLD = 10.0                   # percentage points
STRONG_AICC_DELTA = 6.0
MIN_MATCHED_RESIDUES_FOR_GLOBAL_CHECK = 8

# Generate per-residue plots
MAKE_RESIDUE_PLOTS = True
N_FIT_LINE_POINTS = 400

# Plotting policy for the final clean plots:
# - zero-plateau is the default kinetic model;
# - a free positive plateau is shown only when it is strongly supported;
# - the fixed-100 model is retained in the CSV model comparison but is not
#   drawn in the final clean residue plots.
PLOT_STRONG_PLATEAU_DELTA_AICC = 6.0
PLOT_MIN_SUPPORTED_PLATEAU = 10.0
SHOW_LOW_SNR_POINTS = True
SHOW_MODEL_COMPARISON_PLOTS = False

CONDITION_ORDER = ["WT", "MUT"]
CONDITION_COLORS = {
    "WT": "dodgerblue",
    "MUT": "orangered",
}

CONDITIONS = {
    "WT": {
        "folder_keywords": ["wt"],
        "main_keywords": ["main"],
    },
    "MUT": {
        "folder_keywords": ["mut"],
        "main_keywords": ["main"],
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def intensity_correction_factor(time_value: float) -> float:
    """
    Original acquisition correction:
        time < 13       -> divide by 1
        13 <= time < 22 -> divide by 2
        time >= 22      -> divide by 4
    """
    if time_value >= 22:
        return 4.0
    if time_value >= 13:
        return 2.0
    return 1.0


def extract_zip_if_needed(input_path: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extracted = work_dir / "extracted_input"
        if extracted.exists():
            shutil.rmtree(extracted)
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extracted)
        return extracted

    if input_path.is_dir():
        return input_path

    raise FileNotFoundError(f"Input path not found: {input_path.resolve()}")


def find_condition_folder(root: Path, keywords: Iterable[str]) -> Path:
    candidates = [p for p in root.rglob("*") if p.is_dir()]

    for keyword in keywords:
        matches = [
            p for p in candidates
            if keyword.lower() in p.name.lower()
        ]
        if matches:
            # Prefer the shortest path, as in the original script.
            return sorted(matches, key=lambda p: (len(str(p)), str(p)))[0]

    raise FileNotFoundError(
        f"Could not find a condition folder using keywords: {list(keywords)}"
    )


def find_main_file(folder: Path, main_keywords: Iterable[str]) -> Path:
    files = sorted(folder.glob("*.xlsx"))

    for keyword in main_keywords:
        matches = [
            p for p in files
            if keyword.lower() in p.stem.lower()
        ]
        if matches:
            if len(matches) > 1:
                warnings.warn(
                    f"Multiple possible main files in {folder}: "
                    f"{[p.name for p in matches]}. Using {matches[0].name}."
                )
            return matches[0]

    raise FileNotFoundError(f"Could not find a main/reference file in {folder}")


def parse_time_from_filename(path: Path) -> Optional[float]:
    try:
        return float(path.stem)
    except ValueError:
        return None


def read_ccpn_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name="Table")
    except ValueError:
        return pd.read_excel(path, sheet_name=0)


def parse_assignment(
    value,
) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    if pd.isna(value):
        return None, None, None, None

    text = str(value).strip()
    parts = text.split(".")

    if len(parts) >= 4:
        chain = parts[0]
        try:
            residue_number = int(float(parts[1]))
        except ValueError:
            residue_number = None
        return chain, residue_number, parts[2], parts[3]

    match = re.search(
        r"([A-Za-z0-9]+)\.(\d+)\.([A-Za-z]{3})\.([A-Za-z0-9]+)",
        text,
    )
    if match:
        return (
            match.group(1),
            int(match.group(2)),
            match.group(3),
            match.group(4),
        )

    return None, None, None, None


def add_residue_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    assignment_cols = [
        c for c in ["Assign F1", "Assign F2"]
        if c in df.columns
    ]
    if not assignment_cols:
        raise ValueError(
            "No assignment columns found. Expected 'Assign F1' and/or 'Assign F2'."
        )

    parsed_rows = []
    for _, row in df.iterrows():
        parsed = (None, None, None, None)
        used_assignment = None

        for column in assignment_cols:
            candidate = parse_assignment(row[column])
            if candidate[1] is not None and candidate[2] is not None:
                parsed = candidate
                used_assignment = row[column]
                break

        parsed_rows.append((*parsed, used_assignment))

    parsed_df = pd.DataFrame(
        parsed_rows,
        columns=[
            "chain",
            "residue_number",
            "residue_name",
            "atom_name",
            "assignment_used",
        ],
        index=df.index,
    )

    df = pd.concat([df, parsed_df], axis=1)
    df = df.dropna(subset=["residue_number", "residue_name"]).copy()

    df["residue_number"] = df["residue_number"].astype(int)
    df["residue_name"] = df["residue_name"].astype(str)
    df["chain"] = df["chain"].fillna("NA").astype(str)

    df["residue_id"] = (
        df["chain"]
        + "_"
        + df["residue_number"].astype(str)
        + "_"
        + df["residue_name"]
    )

    return df


def collapse_duplicate_residues(
    df: pd.DataFrame,
    intensity_column: str,
) -> pd.DataFrame:
    """
    Duplicate assignments are explicitly counted. Intensities are summed,
    preserving the behavior of the original processing script.
    """
    df = df.copy()

    optional_numeric = [
        "Pos F1",
        "Pos F2",
        "LW F1 (Hz)",
        "LW F2 (Hz)",
        "S/N",
        "Volume",
        "Height",
    ]
    optional_numeric = [
        c for c in optional_numeric
        if c in df.columns and c != intensity_column
    ]

    for column in [intensity_column] + optional_numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    group_columns = ["chain", "residue_number", "residue_name", "residue_id"]

    agg: Dict[str, str] = {
        intensity_column: "sum",
        "assignment_used": "first",
        "atom_name": "first",
    }
    for column in optional_numeric:
        agg[column] = "mean"

    collapsed = (
        df.groupby(group_columns, as_index=False)
        .agg(agg)
        .sort_values(["chain", "residue_number"])
        .reset_index(drop=True)
    )

    duplicate_counts = (
        df.groupby(group_columns)
        .size()
        .rename("n_rows_collapsed")
        .reset_index()
    )

    collapsed = collapsed.merge(
        duplicate_counts,
        on=group_columns,
        how="left",
    )
    collapsed["duplicate_assignment"] = collapsed["n_rows_collapsed"] > 1

    return collapsed


# ============================================================
# PROCESSING AND NORMALIZATION
# ============================================================

def process_condition(
    condition: str,
    folder: Path,
    main_file: Path,
    output_csv_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process one condition using that condition's own main/unexchanged HNH file.

    Returns
    -------
    long_df
        Processed time-point observations.
    main
        Processed main/reference table.
    duplicate_qc
        Duplicate-assignment summary for main and time-point files.
    """
    duplicate_records: List[pd.DataFrame] = []

    main_raw = add_residue_columns(read_ccpn_table(main_file))
    if INTENSITY_COLUMN not in main_raw.columns:
        raise ValueError(f"{INTENSITY_COLUMN!r} not found in {main_file.name}")

    main = collapse_duplicate_residues(main_raw, INTENSITY_COLUMN)
    main["condition"] = condition
    main["table_type"] = "main"
    main["source_file"] = main_file.name

    duplicate_records.append(
        main[
            [
                "condition",
                "table_type",
                "source_file",
                "residue_id",
                "n_rows_collapsed",
                "duplicate_assignment",
            ]
        ].copy()
    )

    main = main.rename(
        columns={INTENSITY_COLUMN: f"raw_main_{INTENSITY_COLUMN}"}
    )
    main[f"main_{INTENSITY_COLUMN}"] = (
        main[f"raw_main_{INTENSITY_COLUMN}"] / MAIN_INTENSITY_DIVISOR
    )
    main["main_nonpositive"] = (
        ~np.isfinite(main[f"main_{INTENSITY_COLUMN}"])
        | (main[f"main_{INTENSITY_COLUMN}"] <= 0)
    )

    time_files: List[Tuple[float, Path]] = []
    for path in sorted(folder.glob("*.xlsx")):
        if path == main_file:
            continue
        time_value = parse_time_from_filename(path)
        if time_value is not None and time_value <= MAX_TIME_FOR_PROCESSING:
            time_files.append((time_value, path))

    if not time_files:
        raise ValueError(
            f"No numeric time-point files <= {MAX_TIME_FOR_PROCESSING} h "
            f"found in {folder}"
        )

    clean_parts: List[pd.DataFrame] = []

    for time_value, file_path in sorted(time_files, key=lambda item: item[0]):
        raw = add_residue_columns(read_ccpn_table(file_path))
        if INTENSITY_COLUMN not in raw.columns:
            raise ValueError(
                f"{INTENSITY_COLUMN!r} not found in {file_path.name}"
            )

        frame = collapse_duplicate_residues(raw, INTENSITY_COLUMN)
        frame["condition"] = condition
        frame["table_type"] = "timepoint"
        frame["source_file"] = file_path.name

        duplicate_records.append(
            frame[
                [
                    "condition",
                    "table_type",
                    "source_file",
                    "residue_id",
                    "n_rows_collapsed",
                    "duplicate_assignment",
                ]
            ].copy()
        )

        frame = frame.rename(columns={INTENSITY_COLUMN: "raw_intensity"})

        factor = intensity_correction_factor(time_value)
        frame["correction_factor"] = factor
        frame["intensity"] = frame["raw_intensity"] / factor
        frame["time"] = time_value

        clean_parts.append(frame)

    long_df = pd.concat(clean_parts, ignore_index=True)

    merge_columns = [
        "chain",
        "residue_number",
        "residue_name",
        "residue_id",
        f"raw_main_{INTENSITY_COLUMN}",
        f"main_{INTENSITY_COLUMN}",
        "main_nonpositive",
    ]

    long_df = long_df.merge(
        main[merge_columns],
        on=["chain", "residue_number", "residue_name", "residue_id"],
        how="left",
        validate="many_to_one",
    )

    long_df["main_missing"] = long_df[f"main_{INTENSITY_COLUMN}"].isna()

    long_df["percent_intensity"] = (
        long_df["intensity"]
        / long_df[f"main_{INTENSITY_COLUMN}"]
        * 100.0
    )

    # Independently recompute the exact expected result from raw values.
    long_df["percent_intensity_recomputed"] = (
        (
            long_df["raw_intensity"]
            / long_df["correction_factor"]
        )
        / (
            long_df[f"raw_main_{INTENSITY_COLUMN}"]
            / MAIN_INTENSITY_DIVISOR
        )
        * 100.0
    )

    long_df["normalization_abs_error"] = np.abs(
        long_df["percent_intensity"]
        - long_df["percent_intensity_recomputed"]
    )

    if "S/N" in long_df.columns:
        long_df["below_detection"] = (
            long_df["S/N"].notna()
            & (long_df["S/N"] < SNR_THRESHOLD)
        )
    else:
        long_df["below_detection"] = False

    long_df["nonpositive_intensity"] = (
        ~np.isfinite(long_df["percent_intensity"])
        | (long_df["percent_intensity"] <= 0)
    )

    ordered_columns = [
        "condition",
        "time",
        "chain",
        "residue_number",
        "residue_name",
        "residue_id",
        "assignment_used",
        "source_file",
        "raw_intensity",
        "correction_factor",
        "intensity",
        f"raw_main_{INTENSITY_COLUMN}",
        f"main_{INTENSITY_COLUMN}",
        "percent_intensity",
        "percent_intensity_recomputed",
        "normalization_abs_error",
        "S/N" if "S/N" in long_df.columns else None,
        "below_detection",
        "main_missing",
        "main_nonpositive",
        "nonpositive_intensity",
        "n_rows_collapsed",
        "duplicate_assignment",
    ]
    ordered_columns = [
        c for c in ordered_columns
        if c is not None and c in long_df.columns
    ]
    remaining = [
        c for c in long_df.columns
        if c not in ordered_columns
    ]

    long_df = (
        long_df[ordered_columns + remaining]
        .sort_values(["residue_number", "time"])
        .reset_index(drop=True)
    )

    main.to_csv(
        output_csv_dir / f"{safe_name(condition)}_main_reference_clean.csv",
        index=False,
    )
    long_df.to_csv(
        output_csv_dir / f"{safe_name(condition)}_clean_long.csv",
        index=False,
    )

    duplicate_qc = pd.concat(duplicate_records, ignore_index=True)

    return long_df, main, duplicate_qc


# ============================================================
# ROBUST FITTING
# ============================================================

def model_free_plateau(
    t: np.ndarray,
    amplitude: float,
    k: float,
    plateau: float,
) -> np.ndarray:
    return plateau + amplitude * np.exp(-k * t)


def model_zero_plateau(
    t: np.ndarray,
    amplitude: float,
    k: float,
) -> np.ndarray:
    return amplitude * np.exp(-k * t)


def model_fixed_100_zero(
    t: np.ndarray,
    k: float,
) -> np.ndarray:
    return 100.0 * np.exp(-k * t)


MODEL_SPECS = {
    "free_plateau": {
        "function": model_free_plateau,
        "n_parameters": 3,
        "lower": np.array([0.0, 1e-8, 0.0]),
        "upper": np.array([MAX_AMPLITUDE, MAX_K, MAX_PLATEAU]),
    },
    "zero_plateau": {
        "function": model_zero_plateau,
        "n_parameters": 2,
        "lower": np.array([0.0, 1e-8]),
        "upper": np.array([MAX_AMPLITUDE, MAX_K]),
    },
    "fixed_100_zero": {
        "function": model_fixed_100_zero,
        "n_parameters": 1,
        "lower": np.array([1e-8]),
        "upper": np.array([MAX_K]),
    },
}


def make_initial_guess(
    model_name: str,
    t: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    y_max = float(np.nanmax(y))
    y_min = float(np.nanmin(y))

    # A mild default rate. It is only the optimizer's starting point.
    k_guess = 0.03

    if model_name == "free_plateau":
        late_cutoff = np.quantile(t, 0.75)
        late_values = y[t >= late_cutoff]
        plateau_guess = (
            float(np.nanmedian(late_values))
            if late_values.size
            else max(y_min, 0.0)
        )
        plateau_guess = float(np.clip(plateau_guess, 0.0, MAX_PLATEAU))
        amplitude_guess = float(
            np.clip(y_max - plateau_guess, 1.0, MAX_AMPLITUDE)
        )
        return np.array([amplitude_guess, k_guess, plateau_guess])

    if model_name == "zero_plateau":
        amplitude_guess = float(
            np.clip(y_max, 1.0, MAX_AMPLITUDE)
        )
        return np.array([amplitude_guess, k_guess])

    if model_name == "fixed_100_zero":
        return np.array([k_guess])

    raise KeyError(model_name)


def calculate_aicc(sse: float, n: int, p: int) -> float:
    if n <= p + 1 or not np.isfinite(sse):
        return np.inf

    sse = max(float(sse), np.finfo(float).tiny)
    aic = n * np.log(sse / n) + 2 * p
    correction = (2 * p * (p + 1)) / (n - p - 1)
    return float(aic + correction)


def calculate_r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def robust_fit(
    t: np.ndarray,
    y: np.ndarray,
    model_name: str,
) -> Optional[Dict]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t = t[valid]
    y = y[valid]

    if t.size < MIN_POINTS_FOR_FIT:
        return None

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    spec = MODEL_SPECS[model_name]
    function = spec["function"]
    p0 = make_initial_guess(model_name, t, y)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        return function(t, *parameters) - y

    try:
        result = least_squares(
            residuals,
            x0=p0,
            bounds=(spec["lower"], spec["upper"]),
            loss=ROBUST_LOSS,
            f_scale=ROBUST_F_SCALE,
            max_nfev=30000,
        )
    except Exception:
        return None

    if not result.success:
        return None

    parameters = result.x
    y_pred = function(t, *parameters)
    raw_residuals = y - y_pred
    sse = float(np.sum(raw_residuals ** 2))
    rmse = float(np.sqrt(np.mean(raw_residuals ** 2)))
    aicc = calculate_aicc(
        sse,
        n=t.size,
        p=spec["n_parameters"],
    )

    if t.size >= 3:
        residual_time_rho, residual_time_p = spearmanr(t, raw_residuals)
    else:
        residual_time_rho, residual_time_p = np.nan, np.nan

    output: Dict[str, object] = {
        "model": model_name,
        "fit_success": True,
        "n_points": int(t.size),
        "sse": sse,
        "rmse": rmse,
        "r2": calculate_r2(y, y_pred),
        "aicc": aicc,
        "residual_time_spearman_rho": residual_time_rho,
        "residual_time_spearman_p": residual_time_p,
        "t": t,
        "y": y,
        "y_pred": y_pred,
        "parameters": parameters,
    }

    if model_name == "free_plateau":
        amplitude, k, plateau = parameters
    elif model_name == "zero_plateau":
        amplitude, k = parameters
        plateau = 0.0
    else:
        (k,) = parameters
        amplitude = 100.0
        plateau = 0.0

    output.update(
        {
            "amplitude": float(amplitude),
            "k": float(k),
            "plateau": float(plateau),
            "half_life": (
                float(np.log(2.0) / k)
                if k > 0
                else np.nan
            ),
        }
    )

    return output


def fit_all_residues(
    combined: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows: List[Dict] = []
    point_rows: List[pd.DataFrame] = []

    group_columns = [
        "condition",
        "residue_number",
        "residue_name",
        "residue_id",
    ]

    for keys, group in combined.groupby(group_columns, sort=True):
        condition, residue_number, residue_name, residue_id = keys
        group = group.sort_values("time").copy()

        usable = (
            np.isfinite(group["time"])
            & np.isfinite(group["percent_intensity"])
            & (group["percent_intensity"] > 0)
            & (~(group["below_detection"] & EXCLUDE_LOW_SNR_FROM_FIT))
            & (~group["main_missing"])
            & (~group["main_nonpositive"])
        )

        group["fit_used"] = usable

        t = group.loc[usable, "time"].to_numpy(dtype=float)
        y = group.loc[usable, "percent_intensity"].to_numpy(dtype=float)

        model_results: Dict[str, Optional[Dict]] = {}

        for model_name in MODEL_SPECS:
            result = robust_fit(t, y, model_name)
            model_results[model_name] = result

            base = {
                "condition": condition,
                "residue_number": residue_number,
                "residue_name": residue_name,
                "residue_id": residue_id,
                "model": model_name,
                "n_total_points": int(len(group)),
                "n_fit_used": int(usable.sum()),
                "n_below_detection": int(group["below_detection"].sum()),
                "fit_success": result is not None,
            }

            if result is None:
                base.update(
                    {
                        "amplitude": np.nan,
                        "k": np.nan,
                        "plateau": np.nan,
                        "half_life": np.nan,
                        "sse": np.nan,
                        "rmse": np.nan,
                        "r2": np.nan,
                        "aicc": np.inf,
                        "residual_time_spearman_rho": np.nan,
                        "residual_time_spearman_p": np.nan,
                    }
                )
            else:
                for key in [
                    "amplitude",
                    "k",
                    "plateau",
                    "half_life",
                    "sse",
                    "rmse",
                    "r2",
                    "aicc",
                    "residual_time_spearman_rho",
                    "residual_time_spearman_p",
                ]:
                    base[key] = result[key]

            fit_rows.append(base)

        successful = {
            name: result
            for name, result in model_results.items()
            if result is not None and np.isfinite(result["aicc"])
        }

        if successful:
            preferred_model = min(
                successful,
                key=lambda name: successful[name]["aicc"],
            )
            best_aicc = successful[preferred_model]["aicc"]
        else:
            preferred_model = None
            best_aicc = np.nan

        group["preferred_model"] = preferred_model
        point_rows.append(group)

        # Add model-comparison columns to the corresponding fit rows later.
        for row in fit_rows[-len(MODEL_SPECS):]:
            row["preferred_model"] = preferred_model
            if preferred_model is None or not np.isfinite(row["aicc"]):
                row["delta_aicc"] = np.nan
            else:
                row["delta_aicc"] = float(row["aicc"] - best_aicc)

    fits = pd.DataFrame(fit_rows)
    points = pd.concat(point_rows, ignore_index=True)

    return fits, points


# ============================================================
# GLOBAL WT-vs-MUT DIAGNOSTICS
# ============================================================

def matched_condition_diagnostics(
    combined: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wide = (
        combined.pivot_table(
            index=["residue_id", "residue_number", "residue_name", "time"],
            columns="condition",
            values="percent_intensity",
            aggfunc="mean",
        )
        .reset_index()
    )

    if not {"WT", "MUT"}.issubset(wide.columns):
        return wide, pd.DataFrame()

    matched = wide.dropna(subset=["WT", "MUT"]).copy()
    matched = matched[(matched["WT"] > 0) & (matched["MUT"] > 0)].copy()

    matched["MUT_minus_WT"] = matched["MUT"] - matched["WT"]
    matched["MUT_WT_ratio"] = matched["MUT"] / matched["WT"]
    matched["log_MUT_WT_ratio"] = np.log(matched["MUT_WT_ratio"])

    summary_rows: List[Dict] = []

    for time_value, group in matched.groupby("time", sort=True):
        wt = group["WT"].to_numpy(dtype=float)
        mut = group["MUT"].to_numpy(dtype=float)
        ratios = group["MUT_WT_ratio"].to_numpy(dtype=float)
        log_ratios = group["log_MUT_WT_ratio"].to_numpy(dtype=float)
        differences = group["MUT_minus_WT"].to_numpy(dtype=float)

        row: Dict[str, object] = {
            "time": time_value,
            "n_matched_residues": int(len(group)),
            "median_WT": float(np.median(wt)),
            "median_MUT": float(np.median(mut)),
            "median_difference": float(np.median(differences)),
            "difference_mad": robust_mad(differences),
            "median_ratio": float(np.median(ratios)),
            "ratio_q25": float(np.quantile(ratios, 0.25)),
            "ratio_q75": float(np.quantile(ratios, 0.75)),
            "median_log_ratio": float(np.median(log_ratios)),
            "log_ratio_mad": robust_mad(log_ratios),
        }

        if len(group) >= 3 and np.ptp(wt) > 0:
            try:
                slope, intercept, low_slope, high_slope = theilslopes(mut, wt)
            except Exception:
                slope = intercept = low_slope = high_slope = np.nan
        else:
            slope = intercept = low_slope = high_slope = np.nan

        row.update(
            {
                "theilsen_slope": slope,
                "theilsen_intercept": intercept,
                "theilsen_slope_low": low_slope,
                "theilsen_slope_high": high_slope,
            }
        )

        enough = len(group) >= MIN_MATCHED_RESIDUES_FOR_GLOBAL_CHECK

        multiplicative_flag = bool(
            enough
            and abs(row["median_ratio"] - 1.0)
            >= GLOBAL_RATIO_DEVIATION_THRESHOLD
            and row["log_ratio_mad"]
            <= GLOBAL_LOG_RATIO_MAD_THRESHOLD
        )

        additive_flag = bool(
            enough
            and np.isfinite(intercept)
            and abs(intercept) >= ADDITIVE_OFFSET_THRESHOLD
            and np.isfinite(slope)
            and 0.8 <= slope <= 1.2
        )

        row["possible_global_multiplicative_offset"] = multiplicative_flag
        row["possible_global_additive_offset"] = additive_flag

        summary_rows.append(row)

    return matched, pd.DataFrame(summary_rows)


def late_time_floor_summary(
    combined: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    available_times = np.sort(combined["time"].dropna().unique())

    if available_times.size == 0:
        return pd.DataFrame(), pd.DataFrame(), np.nan

    if LATE_TIME_START is None:
        late_start = float(np.quantile(available_times, 0.75))
    else:
        late_start = float(LATE_TIME_START)

    late = combined[combined["time"] >= late_start].copy()

    per_residue = (
        late.groupby(
            ["condition", "residue_id", "residue_number", "residue_name"],
            as_index=False,
        )
        .agg(
            median_late_percent=("percent_intensity", "median"),
            min_late_percent=("percent_intensity", "min"),
            max_late_percent=("percent_intensity", "max"),
            n_late_points=("percent_intensity", "count"),
            n_below_detection=("below_detection", "sum"),
        )
    )

    summary = (
        per_residue.groupby("condition", as_index=False)
        .agg(
            n_residues=("residue_id", "nunique"),
            median_late_floor=("median_late_percent", "median"),
            q25_late_floor=(
                "median_late_percent",
                lambda x: np.quantile(x, 0.25),
            ),
            q75_late_floor=(
                "median_late_percent",
                lambda x: np.quantile(x, 0.75),
            ),
            fraction_residues_above_floor_threshold=(
                "median_late_percent",
                lambda x: np.mean(np.asarray(x) >= LATE_FLOOR_THRESHOLD),
            ),
        )
    )

    summary["possible_common_positive_floor"] = (
        summary["median_late_floor"] >= LATE_FLOOR_THRESHOLD
    )

    return per_residue, summary, late_start


# ============================================================
# PLOTS
# ============================================================

def plot_global_median_trajectories(
    combined: pd.DataFrame,
    out_path: Path,
) -> None:
    summary = (
        combined.groupby(["condition", "time"])["percent_intensity"]
        .agg(
            median="median",
            q25=lambda x: np.quantile(x.dropna(), 0.25),
            q75=lambda x: np.quantile(x.dropna(), 0.75),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for condition in CONDITION_ORDER:
        sub = summary[summary["condition"] == condition].sort_values("time")
        if sub.empty:
            continue
        color = CONDITION_COLORS.get(condition, "black")
        x = sub["time"].to_numpy(dtype=float)
        median = sub["median"].to_numpy(dtype=float)
        q25 = sub["q25"].to_numpy(dtype=float)
        q75 = sub["q75"].to_numpy(dtype=float)

        ax.plot(x, median, marker="o", label=condition, color=color)
        ax.fill_between(x, q25, q75, color=color, alpha=0.15)

    ax.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.5)
    ax.axhline(100, linestyle="--", linewidth=1, color="gray", alpha=0.5)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Percent intensity")
    ax.set_title("Global median trajectory with interquartile range")
    ax.set_xlim(0, PLOT_X_AXIS_MAX)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_ratio_summary(
    time_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    if time_summary.empty:
        return

    frame = time_summary.sort_values("time")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = frame["time"].to_numpy(dtype=float)
    median = frame["median_ratio"].to_numpy(dtype=float)
    q25 = frame["ratio_q25"].to_numpy(dtype=float)
    q75 = frame["ratio_q75"].to_numpy(dtype=float)

    ax.plot(x, median, marker="o")
    ax.fill_between(x, q25, q75, alpha=0.2)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("MUT / WT percent-intensity ratio")
    ax.set_title("Matched-residue global ratio check")
    ax.set_xlim(0, PLOT_X_AXIS_MAX)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_additive_offset_summary(
    time_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    if time_summary.empty:
        return

    frame = time_summary.sort_values("time")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(
        frame["time"],
        frame["theilsen_intercept"],
        marker="o",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    ax.axhline(
        ADDITIVE_OFFSET_THRESHOLD,
        linestyle=":",
        linewidth=1,
        color="gray",
    )
    ax.axhline(
        -ADDITIVE_OFFSET_THRESHOLD,
        linestyle=":",
        linewidth=1,
        color="gray",
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Theil-Sen intercept (percentage points)")
    ax.set_title("Matched-residue additive-offset check")
    ax.set_xlim(0, PLOT_X_AXIS_MAX)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_late_floor_distribution(
    per_residue_late: pd.DataFrame,
    late_start: float,
    out_path: Path,
) -> None:
    if per_residue_late.empty:
        return

    data = []
    labels = []

    for condition in CONDITION_ORDER:
        values = per_residue_late.loc[
            per_residue_late["condition"] == condition,
            "median_late_percent",
        ].dropna().to_numpy(dtype=float)

        if values.size:
            data.append(values)
            labels.append(condition)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.axhline(
        LATE_FLOOR_THRESHOLD,
        linestyle="--",
        linewidth=1,
        color="gray",
    )
    ax.set_ylabel("Median late-time percent intensity")
    ax.set_title(f"Late-time floors from {late_start:g} h onward")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_residue_fits(
    residue_df: pd.DataFrame,
    fits: pd.DataFrame,
    out_path: Path,
) -> None:
    if residue_df.empty:
        return

    residue_number = residue_df["residue_number"].iloc[0]
    residue_name = residue_df["residue_name"].iloc[0]
    residue_id = residue_df["residue_id"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    all_visible_y: List[float] = []

    for condition in CONDITION_ORDER:
        sub = residue_df[
            residue_df["condition"] == condition
        ].sort_values("time")

        if sub.empty:
            continue

        color = CONDITION_COLORS.get(condition, "black")
        detected = sub[~sub["below_detection"]]
        below = sub[sub["below_detection"]]

        ax.scatter(
            detected["time"],
            detected["percent_intensity"],
            s=34,
            color=color,
            label=f"{condition} detected",
            zorder=4,
        )

        if not below.empty:
            ax.scatter(
                below["time"],
                below["percent_intensity"],
                s=44,
                marker="x",
                color=color,
                label=f"{condition} S/N < {SNR_THRESHOLD:g}",
                zorder=5,
            )

        all_visible_y.extend(
            sub["percent_intensity"].dropna().tolist()
        )

        fit_sub = fits[
            (fits["residue_id"] == residue_id)
            & (fits["condition"] == condition)
            & (fits["fit_success"])
        ].copy()

        if fit_sub.empty:
            continue

        max_observed_time = float(sub["time"].max())
        t_line = np.linspace(
            0.0,
            max(MAX_TIME_FOR_PROCESSING, max_observed_time),
            N_FIT_LINE_POINTS,
        )

        for model_name, linestyle, linewidth in [
            ("free_plateau", "--", 1.8),
            ("zero_plateau", "-", 2.0),
            ("fixed_100_zero", ":", 1.4),
        ]:
            row = fit_sub[fit_sub["model"] == model_name]
            if row.empty:
                continue

            row = row.iloc[0]
            function = MODEL_SPECS[model_name]["function"]

            if model_name == "free_plateau":
                parameters = [
                    row["amplitude"],
                    row["k"],
                    row["plateau"],
                ]
            elif model_name == "zero_plateau":
                parameters = [row["amplitude"], row["k"]]
            else:
                parameters = [row["k"]]

            y_line = function(t_line, *parameters)
            ax.plot(
                t_line,
                y_line,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.9,
                label=f"{condition} {model_name}",
                zorder=2,
            )
            all_visible_y.extend(y_line[np.isfinite(y_line)].tolist())

    ax.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.5)
    ax.axhline(100, linestyle="--", linewidth=1, color="gray", alpha=0.5)
    ax.set_title(
        f"Residue {residue_number} {residue_name}: model comparison"
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Percent intensity")
    ax.set_xlim(0, PLOT_X_AXIS_MAX)
    ax.set_xticks(np.arange(0, PLOT_X_AXIS_MAX + 1, 10))

    finite_y = np.asarray(all_visible_y, dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size:
        lower = min(0.0, float(np.min(finite_y)))
        upper = max(110.0, float(np.max(finite_y)) * 1.08)
        ax.set_ylim(lower, upper)

    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)



def choose_primary_plot_model(
    fit_sub: pd.DataFrame,
) -> Tuple[Optional[str], bool]:
    """
    Select the model used in the final residue plot.

    Policy
    ------
    1. Use the zero-plateau model by default.
    2. Use the free-plateau model only when it beats the zero-plateau model
       by at least PLOT_STRONG_PLATEAU_DELTA_AICC and its fitted plateau is
       at least PLOT_MIN_SUPPORTED_PLATEAU percentage points.
    3. If the zero-plateau fit failed, fall back to a successful free-plateau
       fit. The fixed-100 model is diagnostic only and is not used here.
    """
    if fit_sub.empty:
        return None, False

    successful = fit_sub[
        fit_sub["fit_success"]
        & np.isfinite(fit_sub["aicc"])
    ].copy()

    zero = successful[successful["model"] == "zero_plateau"]
    free = successful[successful["model"] == "free_plateau"]

    if not zero.empty and not free.empty:
        zero_row = zero.iloc[0]
        free_row = free.iloc[0]

        strong_plateau = bool(
            (free_row["aicc"] + PLOT_STRONG_PLATEAU_DELTA_AICC <= zero_row["aicc"])
            and (free_row["plateau"] >= PLOT_MIN_SUPPORTED_PLATEAU)
        )

        if strong_plateau:
            return "free_plateau", True
        return "zero_plateau", False

    if not zero.empty:
        return "zero_plateau", False

    if not free.empty:
        return "free_plateau", False

    return None, False


def plot_clean_residue_fit(
    residue_df: pd.DataFrame,
    fits: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Produce the final uncluttered residue plot.

    Filled circles are observations used for fitting. Low-S/N observations
    are retained as x symbols but are not used in the fit. The solid curve is
    a robust exponential decay. By default this is the zero-plateau model
    amplitude * exp(-k*t), i.e. an exponential approaching zero, unless a
    positive plateau is strongly supported by the AICc rule above.
    """
    if residue_df.empty:
        return

    residue_number = residue_df["residue_number"].iloc[0]
    residue_name = residue_df["residue_name"].iloc[0]
    residue_id = residue_df["residue_id"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    visible_y: List[float] = []
    stat_lines: List[str] = []

    for condition in CONDITION_ORDER:
        sub = residue_df[
            residue_df["condition"] == condition
        ].sort_values("time").copy()

        if sub.empty:
            continue

        color = CONDITION_COLORS.get(condition, "black")

        valid_reference = (
            (~sub["main_missing"])
            & (~sub["main_nonpositive"])
        )
        positive = np.isfinite(sub["percent_intensity"]) & (
            sub["percent_intensity"] > 0
        )

        fit_points = sub[sub["fit_used"]].copy()
        low_snr = sub[
            valid_reference
            & positive
            & sub["below_detection"]
            & (~sub["fit_used"])
        ].copy()
        invalid = sub[
            (~valid_reference) | (~positive)
        ].copy()

        if not fit_points.empty:
            ax.scatter(
                fit_points["time"],
                fit_points["percent_intensity"],
                s=36,
                color=color,
                label=f"{condition} fit data",
                zorder=4,
            )
            visible_y.extend(
                fit_points["percent_intensity"].dropna().tolist()
            )

        if SHOW_LOW_SNR_POINTS and not low_snr.empty:
            ax.scatter(
                low_snr["time"],
                low_snr["percent_intensity"],
                s=42,
                marker="x",
                linewidths=1.2,
                color=color,
                alpha=0.75,
                label=f"{condition} S/N < {SNR_THRESHOLD:g}",
                zorder=5,
            )
            visible_y.extend(
                low_snr["percent_intensity"].dropna().tolist()
            )

        # Invalid/non-positive values are not meaningful on the normalized
        # intensity axis. Count them in the plot annotation instead of drawing
        # misleading points at zero.
        n_invalid = int(len(invalid))

        fit_sub = fits[
            (fits["residue_id"] == residue_id)
            & (fits["condition"] == condition)
        ].copy()

        model_name, strong_plateau = choose_primary_plot_model(fit_sub)

        if model_name is None:
            stat_lines.append(
                f"{condition}: no fit; used n={len(fit_points)}, invalid n={n_invalid}"
            )
            continue

        row = fit_sub[
            (fit_sub["model"] == model_name)
            & (fit_sub["fit_success"])
        ]
        if row.empty:
            continue
        row = row.iloc[0]

        t_line = np.linspace(0.0, PLOT_X_AXIS_MAX, N_FIT_LINE_POINTS)
        function = MODEL_SPECS[model_name]["function"]

        if model_name == "free_plateau":
            parameters = [row["amplitude"], row["k"], row["plateau"]]
            model_label = (
                f"{condition} supported plateau fit"
                if strong_plateau
                else f"{condition} free-plateau fallback"
            )
            stat_lines.append(
                f"{condition}: plateau fit; k={row['k']:.4g} h⁻¹, "
                f"t½={row['half_life']:.3g} h, plateau={row['plateau']:.1f}%, "
                f"R²={row['r2']:.3f}, n={int(row['n_fit_used'])}, invalid={n_invalid}"
            )
        else:
            parameters = [row["amplitude"], row["k"]]
            model_label = f"{condition} zero-plateau fit"
            stat_lines.append(
                f"{condition}: zero fit; k={row['k']:.4g} h⁻¹, "
                f"t½={row['half_life']:.3g} h, R²={row['r2']:.3f}, "
                f"n={int(row['n_fit_used'])}, invalid={n_invalid}"
            )

        y_line = function(t_line, *parameters)
        ax.plot(
            t_line,
            y_line,
            color=color,
            linewidth=2.3,
            label=model_label,
            zorder=3,
        )
        visible_y.extend(y_line[np.isfinite(y_line)].tolist())

    ax.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.45)
    ax.axhline(100, linestyle=":", linewidth=1, color="gray", alpha=0.55)
    ax.set_title(f"Residue {residue_number} {residue_name}: HDX exponential decay fit")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Corrected intensity (% of unexchanged HNH)")
    ax.set_xlim(0, PLOT_X_AXIS_MAX)
    ax.set_xticks(np.arange(0, PLOT_X_AXIS_MAX + 1, 10))

    finite_y = np.asarray(visible_y, dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size:
        upper = max(110.0, float(np.quantile(finite_y, 0.995)) * 1.10)
        upper = max(upper, float(np.max(finite_y)) * 1.03)
        ax.set_ylim(0.0, upper)
    else:
        ax.set_ylim(0.0, 110.0)

    if stat_lines:
        ax.text(
            0.985,
            0.975,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.88,
            },
        )

    ax.grid(alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        fontsize=7.5,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.73),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def prepare_late_floor_reference_table(
    combined: pd.DataFrame,
    per_residue_late: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the diagnostic expected for an additive raw-intensity background:

        normalized late floor ~ constant / main reference intensity

    A positive relationship between the normalized floor and inverse reference
    intensity supports a baseline/detection-floor explanation.
    """
    main_column = f"main_{INTENSITY_COLUMN}"

    refs = (
        combined.groupby(
            ["condition", "residue_id", "residue_number", "residue_name"],
            as_index=False,
        )
        .agg(
            main_reference_intensity=(main_column, "first"),
            raw_main_reference_intensity=(f"raw_main_{INTENSITY_COLUMN}", "first"),
        )
    )

    table = per_residue_late.merge(
        refs,
        on=["condition", "residue_id", "residue_number", "residue_name"],
        how="left",
        validate="one_to_one",
    )

    table = table[
        np.isfinite(table["main_reference_intensity"])
        & (table["main_reference_intensity"] > 0)
        & np.isfinite(table["median_late_percent"])
    ].copy()

    table["inverse_main_reference"] = 1.0 / table["main_reference_intensity"]
    return table


def plot_late_floor_vs_inverse_reference(
    diagnostic: pd.DataFrame,
    out_path: Path,
) -> None:
    if diagnostic.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for condition in CONDITION_ORDER:
        sub = diagnostic[
            diagnostic["condition"] == condition
        ].dropna(subset=["inverse_main_reference", "median_late_percent"])

        if sub.empty:
            continue

        color = CONDITION_COLORS.get(condition, "black")
        x = sub["inverse_main_reference"].to_numpy(dtype=float)
        y = sub["median_late_percent"].to_numpy(dtype=float)

        if len(sub) >= 3 and np.ptp(x) > 0:
            rho, p_value = spearmanr(x, y)
            try:
                slope, intercept, _, _ = theilslopes(y, x)
                x_line = np.linspace(np.min(x), np.max(x), 200)
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, color=color, linewidth=1.7, alpha=0.8)
            except Exception:
                rho, p_value = np.nan, np.nan
        else:
            rho, p_value = np.nan, np.nan

        label = f"{condition} (ρ={rho:.2f}, p={p_value:.3g})"
        ax.scatter(x, y, s=36, color=color, alpha=0.8, label=label)

    ax.axhline(
        LATE_FLOOR_THRESHOLD,
        linestyle="--",
        linewidth=1,
        color="gray",
        alpha=0.7,
    )
    ax.set_xlabel("Inverse corrected main/reference intensity (1/a.u.)")
    ax.set_ylabel("Median late-time intensity (%)")
    ax.set_title("Does the positive floor scale with inverse reference intensity?")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ============================================================
# TEXT REPORT
# ============================================================

def bool_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def build_report(
    condition_info: pd.DataFrame,
    combined: pd.DataFrame,
    duplicate_qc: pd.DataFrame,
    global_summary: pd.DataFrame,
    late_summary: pd.DataFrame,
    late_start: float,
    fits: pd.DataFrame,
) -> str:
    lines: List[str] = []

    lines.append("HDX HNH FULL QC REPORT")
    lines.append("=" * 72)
    lines.append("")

    lines.append("1. CONDITION-SPECIFIC UNEXCHANGED REFERENCES")
    lines.append("-" * 72)
    for _, row in condition_info.iterrows():
        lines.append(
            f"{row['condition']}: folder={row['folder']}; "
            f"main/reference={row['main_file']}; "
            f"time files={row['n_time_files_used']}"
        )
    lines.append(
        "Interpretation: each condition was processed against its own selected "
        "main/unexchanged HNH file."
    )
    lines.append("")

    lines.append("2. NORMALIZATION FORMULA VERIFICATION")
    lines.append("-" * 72)
    max_error = combined["normalization_abs_error"].max(skipna=True)
    lines.append(
        "Formula checked independently: "
        "(raw time / time correction factor) / "
        "(raw main / main divisor) * 100."
    )
    lines.append(f"Maximum absolute recomputation error: {max_error:.6g}")
    lines.append(
        "PASS" if np.isfinite(max_error) and max_error < 1e-9
        else "WARNING: recomputed values do not exactly match."
    )
    lines.append("")

    lines.append("3. REFERENCE AND DETECTION QC")
    lines.append("-" * 72)
    lines.append(f"Total observations: {len(combined)}")
    lines.append(f"Missing main/reference values: {bool_count(combined, 'main_missing')}")
    lines.append(f"Non-positive main/reference values: {bool_count(combined, 'main_nonpositive')}")
    lines.append(f"Non-positive normalized intensities: {bool_count(combined, 'nonpositive_intensity')}")
    lines.append(
        f"Points with S/N < {SNR_THRESHOLD:g}: {bool_count(combined, 'below_detection')}"
        + (" (excluded from fits)" if EXCLUDE_LOW_SNR_FROM_FIT else " (included in fits)")
    )
    duplicate_rows = duplicate_qc["duplicate_assignment"].fillna(False).sum()
    lines.append(f"Collapsed duplicate residue assignments: {int(duplicate_rows)}")
    lines.append(
        "Note: duplicate intensities are summed, matching the original script. "
        "Inspect duplicate_assignment_qc.csv to confirm this is intended."
    )
    lines.append("")

    lines.append("4. GLOBAL WT-vs-MUT OFFSET CHECK")
    lines.append("-" * 72)
    if global_summary.empty:
        lines.append("No matched WT/MUT residue-time pairs were available.")
    else:
        mult_times = global_summary.loc[
            global_summary["possible_global_multiplicative_offset"],
            "time",
        ].tolist()
        add_times = global_summary.loc[
            global_summary["possible_global_additive_offset"],
            "time",
        ].tolist()
        lines.append(
            "Time points flagged for a possible common multiplicative offset: "
            + (", ".join(map(str, mult_times)) if mult_times else "none")
        )
        lines.append(
            "Time points flagged for a possible common additive offset: "
            + (", ".join(map(str, add_times)) if add_times else "none")
        )
        lines.append(
            "These are heuristic flags. A common ratio across many residues "
            "suggests scaling; slope near 1 with a large intercept suggests "
            "an additive baseline."
        )
    lines.append("")

    lines.append("5. LATE-TIME FLOOR CHECK")
    lines.append("-" * 72)
    lines.append(f"Late-time window starts at {late_start:g} h.")
    if late_summary.empty:
        lines.append("No late-time observations were available.")
    else:
        for _, row in late_summary.iterrows():
            lines.append(
                f"{row['condition']}: median late floor="
                f"{row['median_late_floor']:.3f}%; "
                f"IQR={row['q25_late_floor']:.3f}-"
                f"{row['q75_late_floor']:.3f}%; "
                f"fraction residues >= {LATE_FLOOR_THRESHOLD:g}%="
                f"{row['fraction_residues_above_floor_threshold']:.3f}; "
                f"common-floor flag={bool(row['possible_common_positive_floor'])}"
            )
    lines.append("")

    lines.append("6. KINETIC MODEL COMPARISON")
    lines.append("-" * 72)
    successful = fits[fits["fit_success"]].copy()
    if successful.empty:
        lines.append("No successful fits.")
    else:
        preferred = (
            successful[
                successful["delta_aicc"].fillna(np.inf) == 0
            ]
            .groupby(["condition", "preferred_model"])
            .size()
            .rename("count")
            .reset_index()
        )
        for _, row in preferred.iterrows():
            lines.append(
                f"{row['condition']}: {row['preferred_model']} preferred for "
                f"{int(row['count'])} residue(s)."
            )

        free = successful[successful["model"] == "free_plateau"].copy()
        zero = successful[successful["model"] == "zero_plateau"][
            ["condition", "residue_id", "aicc"]
        ].rename(columns={"aicc": "zero_aicc"})

        free = free.merge(
            zero,
            on=["condition", "residue_id"],
            how="left",
        )
        free["free_vs_zero_delta"] = free["zero_aicc"] - free["aicc"]
        strongly_nonzero = free[
            (free["free_vs_zero_delta"] >= STRONG_AICC_DELTA)
            & (free["plateau"] >= LATE_FLOOR_THRESHOLD)
        ]
        lines.append(
            "Residue-condition curves strongly supporting a positive plateau "
            f"(free plateau better than zero by AICc >= {STRONG_AICC_DELTA:g} "
            f"and fitted plateau >= {LATE_FLOOR_THRESHOLD:g}%): "
            f"{len(strongly_nonzero)}"
        )
    lines.append("")

    lines.append("7. DECISION RULE")
    lines.append("-" * 72)
    lines.append(
        "Do not force all curves to zero solely because a zero asymptote is "
        "theoretical. First inspect whether the positive floor is shared across "
        "many residues, whether it tracks low S/N, and whether a common scaling "
        "or additive offset is flagged. A shared floor is more consistent with "
        "processing/background than residue-specific protection."
    )
    lines.append("")

    return "\n".join(lines)


def choose_residue_output_subdir(residue_df: pd.DataFrame) -> str:
    """Return the output subfolder for a residue plot based on condition presence."""
    conditions_here = set(residue_df["condition"].dropna().astype(str).unique())

    if conditions_here == {"WT", "MUT"}:
        return "common_WT_MUT_overlay"
    if conditions_here == {"WT"}:
        return "WT_only"
    return "MUT_only"


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    csv_dir = OUTPUT_DIR / "csv"
    plot_dir = OUTPUT_DIR / "plots"

    # Store residue plots in folders matching the earlier layout:
    #   common_WT_MUT_overlay
    #   WT_only
    #   MUT_only
    residue_plot_dir = plot_dir / "residue_model_comparisons"
    clean_residue_plot_dir = plot_dir / "residue_primary_fits"

    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for base_dir in [residue_plot_dir, clean_residue_plot_dir]:
        (base_dir / "common_WT_MUT_overlay").mkdir(parents=True, exist_ok=True)
        (base_dir / "WT_only").mkdir(parents=True, exist_ok=True)
        (base_dir / "MUT_only").mkdir(parents=True, exist_ok=True)

    input_root = extract_zip_if_needed(INPUT_PATH, OUTPUT_DIR)

    condition_tables: List[pd.DataFrame] = []
    condition_info_rows: List[Dict] = []
    duplicate_tables: List[pd.DataFrame] = []

    for condition, config in CONDITIONS.items():
        folder = find_condition_folder(
            input_root,
            config["folder_keywords"],
        )
        main_file = find_main_file(
            folder,
            config["main_keywords"],
        )

        long_df, main_df, duplicate_qc = process_condition(
            condition=condition,
            folder=folder,
            main_file=main_file,
            output_csv_dir=csv_dir,
        )

        condition_tables.append(long_df)
        duplicate_tables.append(duplicate_qc)

        condition_info_rows.append(
            {
                "condition": condition,
                "folder": str(folder),
                "main_file": str(main_file),
                "main_file_name": main_file.name,
                "n_time_files_used": int(long_df["source_file"].nunique()),
                "max_time_used": float(long_df["time"].max()),
                "n_residues_in_main": int(main_df["residue_id"].nunique()),
                "n_residues_in_timepoints": int(long_df["residue_id"].nunique()),
            }
        )

    combined = pd.concat(condition_tables, ignore_index=True)
    combined = combined.sort_values(
        ["residue_number", "condition", "time"]
    ).reset_index(drop=True)

    condition_info = pd.DataFrame(condition_info_rows)
    duplicate_qc = pd.concat(duplicate_tables, ignore_index=True)

    combined.to_csv(csv_dir / "combined_clean_long_with_qc.csv", index=False)
    condition_info.to_csv(csv_dir / "condition_and_main_file_summary.csv", index=False)
    duplicate_qc.to_csv(csv_dir / "duplicate_assignment_qc.csv", index=False)

    correction_summary = (
        combined.groupby(
            ["condition", "time", "correction_factor"],
            as_index=False,
        )
        .agg(
            n_observations=("residue_id", "count"),
            n_residues=("residue_id", "nunique"),
        )
    )
    correction_summary.to_csv(
        csv_dir / "acquisition_correction_summary.csv",
        index=False,
    )

    # Fit models without deleting residual outliers.
    fits, points_with_fit_flags = fit_all_residues(combined)
    fits.to_csv(csv_dir / "robust_model_fits.csv", index=False)
    points_with_fit_flags.to_csv(
        csv_dir / "combined_points_with_fit_flags.csv",
        index=False,
    )

    # Matched-condition scaling/additive checks.
    matched, global_summary = matched_condition_diagnostics(combined)
    matched.to_csv(
        csv_dir / "matched_WT_MUT_residue_time_values.csv",
        index=False,
    )
    global_summary.to_csv(
        csv_dir / "global_scaling_and_additive_offset_by_time.csv",
        index=False,
    )

    # Late-time floor checks.
    per_residue_late, late_summary, late_start = late_time_floor_summary(
        combined
    )
    per_residue_late.to_csv(
        csv_dir / "late_time_floor_by_residue.csv",
        index=False,
    )
    late_summary.to_csv(
        csv_dir / "late_time_floor_summary_by_condition.csv",
        index=False,
    )

    late_floor_reference = prepare_late_floor_reference_table(
        combined,
        per_residue_late,
    )
    late_floor_reference.to_csv(
        csv_dir / "late_floor_vs_main_reference.csv",
        index=False,
    )

    # Model comparison table: one row per residue-condition.
    model_wide = fits.pivot_table(
        index=[
            "condition",
            "residue_number",
            "residue_name",
            "residue_id",
            "preferred_model",
        ],
        columns="model",
        values=[
            "aicc",
            "amplitude",
            "k",
            "plateau",
            "half_life",
            "r2",
            "rmse",
        ],
        aggfunc="first",
    )
    model_wide.columns = [
        f"{metric}_{model}"
        for metric, model in model_wide.columns
    ]
    model_wide = model_wide.reset_index()
    model_wide.to_csv(
        csv_dir / "model_comparison_by_residue_condition.csv",
        index=False,
    )

    # Global plots.
    plot_global_median_trajectories(
        combined,
        plot_dir / "global_median_trajectories.png",
    )
    plot_ratio_summary(
        global_summary,
        plot_dir / "matched_residue_ratio_by_time.png",
    )
    plot_additive_offset_summary(
        global_summary,
        plot_dir / "additive_offset_by_time.png",
    )
    plot_late_floor_distribution(
        per_residue_late,
        late_start,
        plot_dir / "late_time_floor_distribution.png",
    )
    plot_late_floor_vs_inverse_reference(
        late_floor_reference,
        plot_dir / "late_floor_vs_inverse_main_reference.png",
    )

    # Final per-residue plots. These use the zero-plateau model by default
    # and show a positive plateau only when strongly supported by AICc.
    if MAKE_RESIDUE_PLOTS:
        for residue_id, residue_df in points_with_fit_flags.groupby("residue_id"):
            subfolder = choose_residue_output_subdir(residue_df)

            clean_out_path = (
                clean_residue_plot_dir
                / subfolder
                / f"residue_{safe_name(residue_id)}_fit.png"
            )
            plot_clean_residue_fit(
                residue_df=residue_df,
                fits=fits,
                out_path=clean_out_path,
            )

            if SHOW_MODEL_COMPARISON_PLOTS:
                comparison_out_path = (
                    residue_plot_dir
                    / subfolder
                    / f"residue_{safe_name(residue_id)}_models.png"
                )
                plot_residue_fits(
                    residue_df=residue_df,
                    fits=fits,
                    out_path=comparison_out_path,
                )

    report = build_report(
        condition_info=condition_info,
        combined=combined,
        duplicate_qc=duplicate_qc,
        global_summary=global_summary,
        late_summary=late_summary,
        late_start=late_start,
        fits=fits,
    )

    report_path = OUTPUT_DIR / "QC_REPORT.txt"
    report_path.write_text(report, encoding="utf-8")

    zip_path = OUTPUT_DIR.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(OUTPUT_DIR), "zip", OUTPUT_DIR)

    print(report)
    print("")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print(f"Output ZIP:    {zip_path.resolve()}")
    print(f"Main report:   {report_path.resolve()}")


def parse_args() -> None:
    """Allow the USER SETTINGS above to be overridden from the command line."""
    import argparse

    global INPUT_PATH, OUTPUT_DIR, SNR_THRESHOLD, EXCLUDE_LOW_SNR_FROM_FIT

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("input", type=Path, nargs="?", default=INPUT_PATH,
                        help="ZIP file or folder containing one sub-folder per condition")
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_DIR,
                        help="output folder (overwritten if it exists)")
    parser.add_argument("--snr-threshold", type=float, default=SNR_THRESHOLD,
                        help="points below this S/N are flagged as below detection")
    parser.add_argument("--include-low-snr", action="store_true",
                        help="keep low-S/N points in the kinetic fits (flag them only)")
    args = parser.parse_args()
    INPUT_PATH, OUTPUT_DIR = args.input, args.output
    SNR_THRESHOLD = args.snr_threshold
    EXCLUDE_LOW_SNR_FROM_FIT = not args.include_low_snr


if __name__ == "__main__":
    parse_args()
    main()
