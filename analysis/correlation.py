"""
Compute gender–occupation correlation (Cramér's V) for Bias in Bios.

Usage:
    python analysis/correlation.py [--split train] [--output results/cramer_v.csv]

Outputs:
    - CSV with one row per occupation: profession, cramer_v, n_male, n_female, pct_female
    - Bar chart saved to results/figures/cramer_v.png
    - Prints summary stats and top-10 most correlated professions

Cramér's V is a symmetric measure of association between two categorical variables,
bounded [0, 1].  For a 2×2 contingency table (gender × in_profession):
    V = sqrt(chi2 / N)
where chi2 is the chi-squared statistic and N is the total sample count.
V = 0 means no association; V = 1 means perfect association.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.dataset import ID2PROFESSION


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def cramers_v(contingency: np.ndarray) -> float:
    """Cramér's V for an r×c contingency table."""
    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    n = contingency.sum()
    r, c = contingency.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def compute_correlation(split: str = "train") -> pd.DataFrame:
    """
    Load Bias in Bios split and compute per-profession gender–occupation Cramér's V.

    Returns a DataFrame sorted by cramer_v descending with columns:
        profession, cramer_v, n_total, n_male, n_female, pct_female
    """
    from datasets import load_dataset
    ds = load_dataset("LabHC/bias_in_bios", split=split)

    profession_ids = list(ds["profession"])
    genders        = list(ds["gender"])          # 0=male, 1=female

    rows = []
    for prof_id, prof_name in id2profession.items():
        in_prof  = np.array([1 if p == prof_id else 0 for p in profession_ids])
        gender   = np.array(genders)

        n_male_in   = int(((in_prof == 1) & (gender == 0)).sum())
        n_female_in = int(((in_prof == 1) & (gender == 1)).sum())
        n_male_out  = int(((in_prof == 0) & (gender == 0)).sum())
        n_female_out= int(((in_prof == 0) & (gender == 1)).sum())

        contingency = np.array([[n_male_in, n_female_in],
                                 [n_male_out, n_female_out]])

        # Skip if either gender is completely absent (can't compute chi2)
        if contingency.min() == 0 or (n_male_in + n_female_in) == 0:
            cv = 0.0
        else:
            cv = cramers_v(contingency)

        n_total = n_male_in + n_female_in
        pct_female = n_female_in / n_total if n_total > 0 else float("nan")

        rows.append({
            "profession":  prof_name,
            "prof_id":     prof_id,
            "cramer_v":    round(cv, 4),
            "n_total":     n_total,
            "n_male":      n_male_in,
            "n_female":    n_female_in,
            "pct_female":  round(pct_female, 3),
        })

    df = pd.DataFrame(rows).sort_values("cramer_v", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cramer_v(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#c0392b" if v > 0.15 else "#2980b9" for v in df["cramer_v"]]
    ax.bar(df["profession"], df["cramer_v"], color=colors)
    ax.axhline(0.15, color="gray", linestyle="--", linewidth=0.8, label="V=0.15 threshold")
    ax.set_xticklabels(df["profession"], rotation=90, fontsize=8)
    ax.set_ylabel("Cramér's V (gender–occupation)")
    ax.set_title("Gender–Occupation Correlation by Profession (Bias in Bios, train split)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure saved → {out_path}")


def plot_pct_female(df: pd.DataFrame, out_path: str) -> None:
    """Horizontal bar chart: % female per profession, sorted."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_sorted = df.sort_values("pct_female", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 10))
    bars = ax.barh(df_sorted["profession"], df_sorted["pct_female"],
                   color=["#e74c3c" if p > 0.5 else "#3498db" for p in df_sorted["pct_female"]])
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("% Female")
    ax.set_title("Gender Distribution by Profession")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="results/cramer_v.csv")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args()

    print(f"Computing gender–occupation Cramér's V on '{args.split}' split...")
    df = compute_correlation(args.split)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"CSV saved → {args.output}\n")

    print("=== Top 10 most gender-correlated occupations ===")
    print(df.head(10).to_string(index=False))
    print(f"\nMedian Cramér's V: {df['cramer_v'].median():.4f}")
    print(f"Mean   Cramér's V: {df['cramer_v'].mean():.4f}")
    high = df[df["cramer_v"] > 0.15]
    print(f"Professions with V > 0.15: {len(high)} / {len(df)}")

    plot_cramer_v(df, os.path.join(args.fig_dir, "cramer_v.png"))
    plot_pct_female(df, os.path.join(args.fig_dir, "pct_female.png"))


if __name__ == "__main__":
    main()
