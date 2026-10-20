#!/usr/bin/env python3
"""
Generate disappearing assigned peak plots and intensity boxplots
from raw CCPN HDX Excel files.

This script creates 6 figures total:

Disappearing / retained assigned peaks:
1. main vs 2.5 h
2. main vs 76 h
3. 2.5 h vs 76 h

Intensity ratio boxplots:
4. main vs 2.5 h
5. main vs 76 h
6. 2.5 h vs 76 h

Important correction
--------------------
For intensity ratio boxplots ONLY:
    use only residue IDs that are common to both WT and MUT
    within the same comparison.

This makes WT and MUT intensity-ratio distributions directly comparable.

Definitions
-----------
For a given comparison "reference vs target":

Retained:
    assigned residue is positive in both reference and target

Disappeared:
    assigned residue is positive in reference
    but absent or non-positive in target

Intensity boxplot:
    for residues present and positive in both spectra,
    plot target_corrected_intensity / reference_corrected_intensity * 100

Correction rules
----------------
Main/reference spectrum:
    corrected intensity = raw intensity / 4

Time-point spectrum:
    time < 13       -> raw intensity / 1
    13 <= time < 22 -> raw intensity / 2
    time >= 22      -> raw intensity / 4
"""

from __future__ import annotations

import re
import zipfile
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_PATH = Path("examples/synthetic_hdx")
OUTPUT_DIR = Path("output/hdx_peak_retention")

INTENSITY_COLUMN = "Height"
MAIN_INTENSITY_DIVISOR = 4.0

TIMEPOINTS_TO_COMPARE = {
    "main": None,
    "2.5 h": 2.5,
    "76 h": 76.0,
}

COMPARISONS = [
    ("main", "2.5 h"),
    ("main", "76 h"),
    ("2.5 h", "76 h"),
]

CONDITION_ORDER = ["WT", "MUT"]

CONDITION_COLORS = {
    "WT": "dodgerblue",
    "MUT": "orangered",
}

# Boxplots are generated only from residues retained in both spectra and
# shared between WT and MUT for the same comparison.
SHOW_INDIVIDUAL_POINTS = True
SHOW_MEAN_DIAMOND = True
RANDOM_JITTER_SEED = 42

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
# HELPERS
# ============================================================

def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def intensity_correction_factor(time_value: Optional[float]) -> float:
    """
    Return correction factor for a spectrum.

    main -> 4
    time < 13 -> 1
    13 <= time < 22 -> 2
    time >= 22 -> 4
    """
    if time_value is None:
        return MAIN_INTENSITY_DIVISOR

    if time_value >= 22:
        return 4.0
    elif time_value >= 13:
        return 2.0
    else:
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

    raise FileNotFoundError(f"Input path not found: {input_path}")


def find_condition_folder(root: Path, keywords: Iterable[str]) -> Path:
    candidates = [p for p in root.rglob("*") if p.is_dir()]

    for kw in keywords:
        matches = [p for p in candidates if kw.lower() in p.name.lower()]
        if matches:
            return sorted(matches, key=lambda p: len(str(p)))[0]

    raise FileNotFoundError(
        f"Could not find condition folder using keywords: {keywords}"
    )


def find_main_file(folder: Path, main_keywords: Iterable[str]) -> Path:
    xlsx_files = sorted(folder.glob("*.xlsx"))

    for kw in main_keywords:
        matches = [p for p in xlsx_files if kw.lower() in p.stem.lower()]
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not find main peaklist in {folder}")


def parse_time_from_filename(path: Path) -> Optional[float]:
    try:
        return float(path.stem)
    except ValueError:
        return None


def find_time_file(folder: Path, target_time: float) -> Path:
    candidates = []

    for p in sorted(folder.glob("*.xlsx")):
        t = parse_time_from_filename(p)

        if t is not None and np.isclose(t, target_time):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find time file for {target_time} in {folder}"
        )

    return candidates[0]


def read_ccpn_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name="Table")
    except ValueError:
        return pd.read_excel(path, sheet_name=0)


# ============================================================
# CCPN ASSIGNMENT PARSING
# ============================================================

def parse_assignment(value) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Expected formats:
        A.9.THR.H
        A.9.THR.N
    """
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

        residue_name = parts[2]
        atom_name = parts[3]

        return chain, residue_number, residue_name, atom_name

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

    assignment_cols = [c for c in ["Assign F1", "Assign F2"] if c in df.columns]

    if not assignment_cols:
        raise ValueError(
            "No assignment columns found. Expected 'Assign F1' and/or 'Assign F2'."
        )

    parsed_rows = []

    for _, row in df.iterrows():
        parsed = (None, None, None, None)
        used_assignment = None

        for col in assignment_cols:
            candidate = parse_assignment(row[col])

            if candidate[1] is not None and candidate[2] is not None:
                parsed = candidate
                used_assignment = row[col]
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
    df = df.dropna(subset=["residue_number", "residue_name"])

    df["residue_number"] = df["residue_number"].astype(int)
    df["residue_name"] = df["residue_name"].astype(str)
    df["residue_id"] = df["residue_number"].astype(str) + "_" + df["residue_name"]

    return df


def collapse_duplicate_residues(
    df: pd.DataFrame,
    intensity_column: str,
) -> pd.DataFrame:
    """
    Collapse multiple assigned peaks for one residue by summing intensity.
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

    for c in [intensity_column] + optional_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = {
        intensity_column: "sum",
        "assignment_used": "first",
        "chain": "first",
        "residue_name": "first",
    }

    for c in optional_numeric:
        agg[c] = "mean"

    collapsed = (
        df.groupby(["residue_number", "residue_id"], as_index=False)
        .agg(agg)
        .sort_values("residue_number")
        .reset_index(drop=True)
    )

    return collapsed


# ============================================================
# SPECTRUM CLEANING
# ============================================================

def clean_spectrum(
    path: Path,
    condition: str,
    spectrum_label: str,
    time_value: Optional[float],
) -> pd.DataFrame:
    raw = read_ccpn_table(path)
    raw = add_residue_columns(raw)

    if INTENSITY_COLUMN not in raw.columns:
        raise ValueError(f"{INTENSITY_COLUMN!r} not found in {path.name}")

    df = collapse_duplicate_residues(raw, INTENSITY_COLUMN)
    df = df.rename(columns={INTENSITY_COLUMN: "raw_intensity"})

    correction_factor = intensity_correction_factor(time_value)

    df["condition"] = condition
    df["spectrum_label"] = spectrum_label
    df["time"] = time_value if time_value is not None else np.nan
    df["source_file"] = path.name
    df["correction_factor"] = correction_factor
    df["corrected_intensity"] = df["raw_intensity"] / correction_factor
    df["is_positive"] = df["corrected_intensity"] > 0

    ordered_cols = [
        "condition",
        "spectrum_label",
        "time",
        "residue_number",
        "residue_name",
        "residue_id",
        "assignment_used",
        "raw_intensity",
        "correction_factor",
        "corrected_intensity",
        "is_positive",
        "source_file",
    ]

    remaining = [c for c in df.columns if c not in ordered_cols]

    return df[ordered_cols + remaining]


def load_condition_spectra(
    root: Path,
    condition: str,
    config: dict,
) -> dict[str, pd.DataFrame]:
    folder = find_condition_folder(root, config["folder_keywords"])
    main_file = find_main_file(folder, config["main_keywords"])

    spectra = {}

    for label, time_value in TIMEPOINTS_TO_COMPARE.items():
        if label == "main":
            path = main_file
        else:
            path = find_time_file(folder, time_value)

        spectra[label] = clean_spectrum(
            path=path,
            condition=condition,
            spectrum_label=label,
            time_value=time_value,
        )

    return spectra


# ============================================================
# COMPARISON LOGIC
# ============================================================

def compare_spectra(
    condition: str,
    reference_label: str,
    target_label: str,
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    comparison = f"{reference_label} vs {target_label}"

    ref = reference_df.copy()
    tgt = target_df.copy()

    ref_positive = ref[ref["is_positive"] == True].copy()
    tgt_positive = tgt[tgt["is_positive"] == True].copy()

    ref_ids = set(ref_positive["residue_id"])
    tgt_ids = set(tgt_positive["residue_id"])

    disappeared_ids = sorted(ref_ids - tgt_ids)
    common_ids = sorted(ref_ids & tgt_ids)
    target_only_ids = sorted(tgt_ids - ref_ids)

    ref_small = ref[
        [
            "residue_number",
            "residue_name",
            "residue_id",
            "assignment_used",
            "corrected_intensity",
            "raw_intensity",
            "is_positive",
        ]
    ].rename(
        columns={
            "corrected_intensity": "reference_corrected_intensity",
            "raw_intensity": "reference_raw_intensity",
            "is_positive": "reference_is_positive",
            "assignment_used": "reference_assignment",
        }
    )

    tgt_small = tgt[
        [
            "residue_number",
            "residue_name",
            "residue_id",
            "assignment_used",
            "corrected_intensity",
            "raw_intensity",
            "is_positive",
        ]
    ].rename(
        columns={
            "corrected_intensity": "target_corrected_intensity",
            "raw_intensity": "target_raw_intensity",
            "is_positive": "target_is_positive",
            "assignment_used": "target_assignment",
        }
    )

    merged = ref_small.merge(
        tgt_small,
        on=["residue_number", "residue_name", "residue_id"],
        how="outer",
    )

    merged["condition"] = condition
    merged["comparison"] = comparison
    merged["reference_label"] = reference_label
    merged["target_label"] = target_label

    merged["present_in_reference"] = merged["residue_id"].isin(ref_ids)
    merged["present_in_target"] = merged["residue_id"].isin(tgt_ids)

    merged["status"] = "other"
    merged.loc[merged["residue_id"].isin(common_ids), "status"] = "present_in_both"
    merged.loc[merged["residue_id"].isin(disappeared_ids), "status"] = "disappeared_from_reference"
    merged.loc[merged["residue_id"].isin(target_only_ids), "status"] = "target_only"

    common_table = merged[merged["status"] == "present_in_both"].copy()

    common_table["target_percent_of_reference"] = (
        common_table["target_corrected_intensity"]
        / common_table["reference_corrected_intensity"]
        * 100
    )

    summary = {
        "condition": condition,
        "comparison": comparison,
        "reference_label": reference_label,
        "target_label": target_label,
        "n_reference_positive_assigned": len(ref_ids),
        "n_target_positive_assigned": len(tgt_ids),
        "n_present_in_both": len(common_ids),
        "n_disappeared_from_reference": len(disappeared_ids),
        "n_target_only": len(target_only_ids),
        "fraction_disappeared_from_reference_percent": (
            len(disappeared_ids) / len(ref_ids) * 100 if len(ref_ids) > 0 else np.nan
        ),
    }

    return merged, common_table, summary


def filter_common_residues_between_conditions_for_intensity(
    common_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For intensity ratio boxplots only:

    Keep only residue_ids that are present in BOTH WT and MUT
    within the same comparison.

    This makes the WT and MUT boxplots comparable because both
    distributions are calculated from the same retained assigned residues.
    """
    filtered_parts = []

    for comparison, comp_df in common_df.groupby("comparison"):
        residue_sets = {}

        for condition in CONDITION_ORDER:
            condition_df = comp_df[comp_df["condition"] == condition]
            residue_sets[condition] = set(condition_df["residue_id"])

        if not all(condition in residue_sets for condition in CONDITION_ORDER):
            continue

        common_residues = set.intersection(
            *[residue_sets[condition] for condition in CONDITION_ORDER]
        )

        filtered = comp_df[comp_df["residue_id"].isin(common_residues)].copy()
        filtered_parts.append(filtered)

    if not filtered_parts:
        return pd.DataFrame(columns=common_df.columns)

    return pd.concat(filtered_parts, ignore_index=True)


# ============================================================
# PLOTTING
# ============================================================

def plot_stacked_retained_lost_for_comparison(
    summary_df: pd.DataFrame,
    comparison: str,
    out_path: Path,
) -> None:
    sub = summary_df[summary_df["comparison"] == comparison].copy()

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6.5))

    x_positions = np.arange(len(CONDITION_ORDER))
    bar_width = 0.6

    for i, condition in enumerate(CONDITION_ORDER):
        row = sub[sub["condition"] == condition]

        if row.empty:
            continue

        row = row.iloc[0]

        total = int(row["n_reference_positive_assigned"])
        retained = int(row["n_present_in_both"])
        lost = int(row["n_disappeared_from_reference"])

        retained_pct = retained / total * 100 if total > 0 else 0
        lost_pct = lost / total * 100 if total > 0 else 0

        retained_color = CONDITION_COLORS.get(condition, "gray")

        ax.bar(
            x_positions[i],
            retained,
            width=bar_width,
            color=retained_color,
            edgecolor="black",
            linewidth=1.6,
        )

        ax.bar(
            x_positions[i],
            lost,
            width=bar_width,
            bottom=retained,
            color="lightgray",
            edgecolor="black",
            linewidth=1.6,
        )

        ax.text(
            x_positions[i],
            total + max(2, total * 0.04),
            f"Total = {total}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

        if retained > 0:
            ax.text(
                x_positions[i],
                retained / 2,
                f"{retained}\n({retained_pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
            )

        if lost > 0:
            ax.text(
                x_positions[i],
                retained + lost / 2,
                f"{lost}\n({lost_pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
            )

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor=CONDITION_COLORS["WT"],
            edgecolor="black",
            linewidth=1.4,
            label="Retained",
        ),
        Patch(
            facecolor="lightgray",
            edgecolor="black",
            linewidth=1.4,
            label="Lost",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=12,
        loc="upper right",
    )

    ax.set_title(
        f"Assigned peak retention: {comparison}",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_ylabel("Number of assigned peaks", fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(CONDITION_ORDER, fontsize=13, fontweight="bold")

    ymax = sub["n_reference_positive_assigned"].max()
    ax.set_ylim(0, ymax * 1.18)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_all_stacked_retained_lost(
    summary_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    for comparison in summary_df["comparison"].drop_duplicates():
        out_path = plot_dir / f"stacked_retained_lost_{safe_name(comparison)}.png"

        plot_stacked_retained_lost_for_comparison(
            summary_df=summary_df,
            comparison=comparison,
            out_path=out_path,
        )


def plot_intensity_ratio_boxplot_for_comparison(
    common_df: pd.DataFrame,
    comparison: str,
    out_path: Path,
) -> None:
    """
    Plot WT and MUT intensity-ratio distributions using only the remaining
    retained residues that are shared between both conditions.

    Every residue contributing to the WT box also contributes to the MUT
    box for the same comparison.
    """
    plot_df = common_df[
        (common_df["comparison"] == comparison)
        & np.isfinite(common_df["target_percent_of_reference"])
        & (common_df["target_percent_of_reference"] > 0)
    ].copy()

    if plot_df.empty:
        return

    # Final defensive pairing check: retain exactly the same residue IDs in
    # WT and MUT. This prevents unequal residue sets after any earlier filter.
    residue_sets = {
        condition: set(
            plot_df.loc[
                plot_df["condition"] == condition,
                "residue_id",
            ]
        )
        for condition in CONDITION_ORDER
    }

    if not all(residue_sets.get(condition) for condition in CONDITION_ORDER):
        return

    shared_ids = set.intersection(
        *[residue_sets[condition] for condition in CONDITION_ORDER]
    )
    plot_df = plot_df[plot_df["residue_id"].isin(shared_ids)].copy()

    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    data = []
    labels = []
    colors = []

    for condition in CONDITION_ORDER:
        condition_df = (
            plot_df[plot_df["condition"] == condition]
            .drop_duplicates(subset=["residue_id"])
            .sort_values("residue_id")
        )

        values = condition_df["target_percent_of_reference"].to_numpy(dtype=float)

        if len(values) == 0:
            continue

        data.append(values)
        labels.append(f"{condition}\n(n={len(values)})")
        colors.append(CONDITION_COLORS.get(condition, "gray"))

    if len(data) != len(CONDITION_ORDER):
        plt.close(fig)
        return

    positions = np.arange(1, len(data) + 1)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2.2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
        boxprops={"linewidth": 1.7},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(RANDOM_JITTER_SEED)
    all_visible_values = []

    for pos, values, color in zip(positions, data, colors):
        if SHOW_INDIVIDUAL_POINTS:
            jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=42,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                alpha=0.80,
                zorder=3,
            )

        if SHOW_MEAN_DIAMOND:
            ax.scatter(
                pos,
                float(np.mean(values)),
                marker="D",
                s=80,
                color="black",
                zorder=4,
                label="Mean" if pos == positions[0] else None,
            )

        all_visible_values.extend(values.tolist())

    all_values = np.asarray(all_visible_values, dtype=float)
    data_min = float(np.nanmin(all_values))
    data_max = float(np.nanmax(all_values))
    data_range = data_max - data_min

    if data_range == 0:
        ymin = max(0.0, data_min - 10.0)
        ymax = data_max + 10.0
    else:
        ymin = max(0.0, data_min - 0.08 * data_range)
        ymax = data_max + 0.12 * data_range

    if ymin <= 100.0 <= ymax:
        ax.axhline(
            100.0,
            linestyle="--",
            linewidth=1.2,
            color="gray",
            alpha=0.65,
            label="100% reference",
        )

    ax.set_ylim(ymin, ymax)
    ax.set_title(
        f"Retained assigned-peak intensity: {comparison}",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Target / reference corrected intensity (%)",
        fontsize=12,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")

    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(legend_labels, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            frameon=False,
            fontsize=10,
            loc="upper right",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_all_intensity_ratio_boxplots(
    common_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    """Create one paired-residue boxplot for each requested comparison."""
    for reference_label, target_label in COMPARISONS:
        comparison = f"{reference_label} vs {target_label}"
        out_path = (
            plot_dir
            / f"boxplot_retained_intensity_{safe_name(comparison)}.png"
        )
        plot_intensity_ratio_boxplot_for_comparison(
            common_df=common_df,
            comparison=comparison,
            out_path=out_path,
        )


def summarize_boxplot_data(common_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the exact remaining values used in each boxplot."""
    if common_df.empty:
        return pd.DataFrame()

    frame = common_df[
        np.isfinite(common_df["target_percent_of_reference"])
        & (common_df["target_percent_of_reference"] > 0)
    ].copy()

    if frame.empty:
        return pd.DataFrame()

    return (
        frame.groupby(["comparison", "condition"], as_index=False)
        .agg(
            n_residues=("residue_id", "nunique"),
            mean_percent=("target_percent_of_reference", "mean"),
            median_percent=("target_percent_of_reference", "median"),
            q25_percent=(
                "target_percent_of_reference",
                lambda x: np.quantile(x, 0.25),
            ),
            q75_percent=(
                "target_percent_of_reference",
                lambda x: np.quantile(x, 0.75),
            ),
            minimum_percent=("target_percent_of_reference", "min"),
            maximum_percent=("target_percent_of_reference", "max"),
        )
        .sort_values(["comparison", "condition"])
        .reset_index(drop=True)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    csv_dir = OUTPUT_DIR / "clean_csv"
    plot_dir = OUTPUT_DIR / "plots"

    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    input_root = extract_zip_if_needed(INPUT_PATH, OUTPUT_DIR)

    all_spectra = []
    all_status_tables = []
    all_common_tables = []
    summary_rows = []

    for condition, config in CONDITIONS.items():
        spectra = load_condition_spectra(
            root=input_root,
            condition=condition,
            config=config,
        )

        for label, spectrum_df in spectra.items():
            all_spectra.append(spectrum_df)

            spectrum_df.to_csv(
                csv_dir / f"{safe_name(condition)}_{safe_name(label)}_assigned_clean.csv",
                index=False,
            )

        for reference_label, target_label in COMPARISONS:
            status_table, common_table, summary = compare_spectra(
                condition=condition,
                reference_label=reference_label,
                target_label=target_label,
                reference_df=spectra[reference_label],
                target_df=spectra[target_label],
            )

            all_status_tables.append(status_table)
            all_common_tables.append(common_table)
            summary_rows.append(summary)

    all_spectra_df = pd.concat(all_spectra, ignore_index=True)
    status_df = pd.concat(all_status_tables, ignore_index=True)
    common_df_all_retained = pd.concat(all_common_tables, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    common_df_shared_wt_mut = filter_common_residues_between_conditions_for_intensity(
        common_df_all_retained
    )

    all_spectra_df.to_csv(
        csv_dir / "all_assigned_spectra_clean.csv",
        index=False,
    )

    status_df.to_csv(
        csv_dir / "assigned_peak_disappearance_status_by_residue.csv",
        index=False,
    )

    common_df_all_retained.to_csv(
        csv_dir / "common_assigned_peak_intensity_percentages_all_retained.csv",
        index=False,
    )

    common_df_shared_wt_mut.to_csv(
        csv_dir / "common_assigned_peak_intensity_percentages_shared_WT_MUT.csv",
        index=False,
    )

    boxplot_summary_df = summarize_boxplot_data(
        common_df_shared_wt_mut
    )
    boxplot_summary_df.to_csv(
        csv_dir / "boxplot_summary_shared_retained_residues.csv",
        index=False,
    )

    summary_df.to_csv(
        csv_dir / "disappeared_peak_summary.csv",
        index=False,
    )

    # 3 disappearing plots
    plot_all_stacked_retained_lost(
        summary_df=summary_df,
        plot_dir=plot_dir,
    )

    # 3 intensity boxplots using only shared WT/MUT residues
    plot_all_intensity_ratio_boxplots(
        common_df=common_df_shared_wt_mut,
        plot_dir=plot_dir,
    )

    zip_out = OUTPUT_DIR.with_suffix(".zip")

    if zip_out.exists():
        zip_out.unlink()

    shutil.make_archive(
        str(OUTPUT_DIR),
        "zip",
        OUTPUT_DIR,
    )

    print(f"Done. Output folder: {OUTPUT_DIR}")
    print(f"Output ZIP: {zip_out}")

    print("\nCSV outputs:")
    for p in sorted(csv_dir.glob("*.csv")):
        print(f"  - {p.name}")

    print("\nGenerated plot files:")
    for p in sorted(plot_dir.glob("*.png")):
        print(f"  - {p.name}")

    print("\nImportant intensity CSVs:")
    print("  - common_assigned_peak_intensity_percentages_shared_WT_MUT.csv")
    print("  - boxplot_summary_shared_retained_residues.csv")


def parse_args() -> None:
    """Allow the USER SETTINGS above to be overridden from the command line."""
    import argparse

    global INPUT_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("input", type=Path, nargs="?", default=INPUT_PATH,
                        help="ZIP file or folder containing one sub-folder per condition")
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_DIR,
                        help="output folder (overwritten if it exists)")
    args = parser.parse_args()
    INPUT_PATH, OUTPUT_DIR = args.input, args.output


if __name__ == "__main__":
    parse_args()
    main()
