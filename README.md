# Protein–water dynamics: NMR hydrogen–deuterium exchange analysis

Python workflows for quantifying backbone amide hydrogen–deuterium exchange (HDX)
from series of 2D HN NMR peak tables, comparing a wild-type protein (WT) with a
mutant (MUT). Written for a kinase allostery project, where the question was
whether a point mutation changes how solvent-accessible and dynamic different
regions of the protein are.

The repository contains **code and simulated example data only**. No
experimental data are included.

## What the scripts do

### `hdx_kinetics_qc.py`: normalisation, QC and kinetic fitting

1. **Normalisation.** Each exchanged spectrum is divided by the condition's own
   unexchanged reference spectrum, after correcting for acquisition scaling
   factors. The percent-intensity calculation is recomputed independently as
   a check.
2. **Quality control.**
   - Missing or non-positive reference peaks and duplicate assignments are reported.
   - Points below a signal-to-noise threshold are flagged instead of being
     silently deleted. By default they are left out of the fits; use
     `--include-low-snr` to fit them too.
3. **Kinetic model comparison.** For every residue and condition, three robust
   (soft-L1 loss) exponential models are fitted:

   | model | equation |
   |---|---|
   | free plateau | `plateau + A·exp(−k·t)` |
   | zero plateau | `A·exp(−k·t)` |
   | fixed start | `100·exp(−k·t)` |

   The models are compared by **AICc**. A positive plateau (protected fraction)
   is reported only when the AICc evidence for it is strong. The outputs are
   exchange rates `k`, half-lives, R² and RMSE.
4. **Systematic-offset diagnostics.** Matched WT/MUT residues at each time point
   are tested for a common multiplicative (ratio) or additive (baseline)
   offset, using medians, MAD, Theil–Sen slopes and Spearman correlation. A
   late-time "floor" analysis checks whether residual intensity is shared
   across residues, which points to processing or background, or is
   residue-specific, which points to real protection.
5. **Outputs.** Tidy CSV tables, global and per-residue plots, and a plain-text
   `QC_REPORT.txt`.

### `hdx_peak_retention.py`: retained vs. lost peaks

For chosen pairs of spectra (for example unexchanged vs 2.5 h vs 76 h):

- Counts which assigned peaks are **retained** or **disappear** in WT and MUT.
- Plots intensity-ratio distributions using **only residues shared by WT and
  MUT** within each comparison, so the two distributions can be compared
  directly.

## Quick start

```bash
pip install -r requirements.txt

# 1. create a simulated data set (40 residues, 9 time points, WT + MUT)
python examples/make_synthetic_data.py

# 2. run the analyses (outputs go to ./output)
python hdx_kinetics_qc.py examples/synthetic_hdx
python hdx_peak_retention.py examples/synthetic_hdx

# options
python hdx_kinetics_qc.py --help
python hdx_kinetics_qc.py my_data.zip -o output/run1 --snr-threshold 5 --include-low-snr
```

## Input format

The input is a folder or ZIP file with one sub-folder per condition, found by
keyword (`wt`, `mut`). Each sub-folder contains CCPN peak tables exported to
Excel:

```
my_data/
  WT_data/   main.xlsx  0.5.xlsx  1.xlsx  2.5.xlsx ... 76.xlsx
  MUT_data/  main.xlsx  0.5.xlsx  1.xlsx  2.5.xlsx ... 76.xlsx
```

- `main.xlsx` is the unexchanged reference spectrum.
- Every other file is named by its exchange time in hours.
- Required columns: `Assign F1` / `Assign F2` (e.g. `A.27.LYS.H`) and `Height`.
- `S/N` is optional and is used for detection flags.

Condition names, keywords, thresholds and acquisition scaling factors are set
in the **USER SETTINGS** block at the top of each script.

## How this was developed

The analysis design, the statistical and QC decisions (normalisation strategy,
S/N handling, model choice and AICc criteria, offset diagnostics) and the
validation against experimental spectra are my own. Much of the code was
written iteratively with AI coding assistants, which I directed, tested and
corrected.

## Requirements

Python ≥ 3.9, NumPy, pandas, SciPy, Matplotlib, openpyxl (see `requirements.txt`).
