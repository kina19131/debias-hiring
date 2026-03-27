"""
Fairness metrics: Equalized Opportunity and Equalized Odds per profession.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Per-profession fairness stats
# ---------------------------------------------------------------------------

def compute_equalized_opps(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sensitive_attr: torch.Tensor,
    id2profession: dict,
    epoch: int,
) -> list[dict]:
    """
    Compute per-profession TPR gap (Equal Opportunity).

    Returns a list of dicts with keys:
        epoch, profession, TPR_male, TPR_female, FPR_male, FPR_female,
        TPR_diff, FPR_diff, Odd_gap, Odd_previlaged_gender
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    sensitive_attr = sensitive_attr.detach().cpu().numpy()

    stats = []
    for class_id in sorted(id2profession.keys()):
        profession_name = id2profession[class_id]
        y_true_bin = (y_true == class_id).astype(int)
        y_pred_bin = (y_pred == class_id).astype(int)

        tprs, fprs = {}, {}
        for g in [0, 1]:
            mask = sensitive_attr == g
            if mask.sum() == 0:
                tprs[g] = fprs[g] = np.nan
                continue
            tp = ((y_true_bin[mask] == 1) & (y_pred_bin[mask] == 1)).sum()
            fn = ((y_true_bin[mask] == 1) & (y_pred_bin[mask] == 0)).sum()
            fp = ((y_true_bin[mask] == 0) & (y_pred_bin[mask] == 1)).sum()
            tn = ((y_true_bin[mask] == 0) & (y_pred_bin[mask] == 0)).sum()
            tprs[g] = tp / (tp + fn + 1e-6)
            fprs[g] = fp / (fp + tn + 1e-6)

        if any(np.isnan(v) for v in [tprs[0], tprs[1], fprs[0], fprs[1]]):
            continue

        stats.append({
            "epoch": epoch,
            "profession": profession_name,
            "TPR_male": tprs[0],
            "TPR_female": tprs[1],
            "FPR_male": fprs[0],
            "FPR_female": fprs[1],
            "TPR_diff": abs(tprs[1] - tprs[0]),
            "FPR_diff": abs(fprs[1] - fprs[0]),
            "Odd_gap": abs(tprs[1] - tprs[0]) + abs(fprs[1] - fprs[0]),
            "Odd_previlaged_gender": "female" if (tprs[1] + fprs[1]) > (tprs[0] + fprs[0]) else "male",
        })

    return stats


def compute_equalized_odds(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sensitive_attr: torch.Tensor,
    id2profession: dict,
    epoch: int,
) -> list[dict]:
    """
    Compute per-profession Equalized Odds gap (0.5 * (TPR_gap + FPR_gap)).

    Returns a list of dicts with keys:
        epoch, profession, TPR_male, TPR_female, FPR_male, FPR_female,
        Odd_gap, Odd_previlaged_gender
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    sensitive_attr = sensitive_attr.detach().cpu().numpy()

    stats = []
    for class_id in sorted(id2profession.keys()):
        profession_name = id2profession[class_id]
        y_true_bin = (y_true == class_id).astype(int)
        y_pred_bin = (y_pred == class_id).astype(int)

        tprs, fprs = {}, {}
        for g in [0, 1]:
            mask = sensitive_attr == g
            if mask.sum() == 0:
                tprs[g] = fprs[g] = np.nan
                continue
            tp = ((y_true_bin[mask] == 1) & (y_pred_bin[mask] == 1)).sum()
            fn = ((y_true_bin[mask] == 1) & (y_pred_bin[mask] == 0)).sum()
            fp = ((y_true_bin[mask] == 0) & (y_pred_bin[mask] == 1)).sum()
            tn = ((y_true_bin[mask] == 0) & (y_pred_bin[mask] == 0)).sum()
            tprs[g] = tp / (tp + fn + 1e-6)
            fprs[g] = fp / (fp + tn + 1e-6)

        if any(np.isnan(v) for v in [tprs[0], tprs[1], fprs[0], fprs[1]]):
            continue

        stats.append({
            "epoch": epoch,
            "profession": profession_name,
            "TPR_male": tprs[0],
            "TPR_female": tprs[1],
            "FPR_male": fprs[0],
            "FPR_female": fprs[1],
            "Odd_gap": 0.5 * (abs(tprs[1] - tprs[0]) + abs(fprs[1] - fprs[0])),
            "Odd_previlaged_gender": "female" if (tprs[1] + fprs[1]) > (tprs[0] + fprs[0]) else "male",
        })

    return stats


# ---------------------------------------------------------------------------
# Detailed evaluation + plotting (for post-training / test set analysis)
# ---------------------------------------------------------------------------

def log_confusion_and_fairness(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    id2profession: dict,
    lambda_val: float = 0.0,
    save_dir: str = "results/figures",
) -> dict:
    """
    Run the model on dataloader, print per-profession confusion stats and
    TPR gaps, log per-profession TPR gap to W&B, and save a bar chart.

    Returns a dict with global and per-profession fairness stats.
    """
    model.eval()
    all_preds, all_labels, all_genders = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            genders = batch["gender"].to(device)
            logits, _ = model(input_ids)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(labels.cpu())
            all_genders.append(genders.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    genders = torch.cat(all_genders).numpy()

    df = pd.DataFrame({"label": labels, "pred": preds, "gender": genders})
    df["label_name"] = df["label"].map(id2profession)
    df["pred_name"] = df["pred"].map(id2profession)

    tpr_male = ((df["label"] == df["pred"]) & (df["gender"] == 0)).sum() / max((df["gender"] == 0).sum(), 1)
    tpr_female = ((df["label"] == df["pred"]) & (df["gender"] == 1)).sum() / max((df["gender"] == 1).sum(), 1)
    tpr_gap = abs(tpr_male - tpr_female)
    print(f"\nGlobal TPR — male: {tpr_male:.4f}  female: {tpr_female:.4f}  gap: {tpr_gap:.4f}\n")

    confusion_stats = {}
    profession_tpr_gap = {}

    print("Per-profession confusion metrics")
    print("=" * 55)
    for prof_id in sorted(id2profession.keys()):
        profession = id2profession[prof_id]
        y_true_bin = (df["label"] == prof_id).astype(int)
        y_pred_bin = (df["pred"] == prof_id).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        confusion_stats[profession] = {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}

        male_df = df[(df["gender"] == 0) & (df["label"] == prof_id)]
        female_df = df[(df["gender"] == 1) & (df["label"] == prof_id)]
        male_tpr = (male_df["pred"] == prof_id).sum() / max(len(male_df), 1)
        female_tpr = (female_df["pred"] == prof_id).sum() / max(len(female_df), 1)
        gap = abs(male_tpr - female_tpr)
        profession_tpr_gap[profession] = gap

        wandb.log({f"Test_TPR_gap/{profession}": gap})
        print(f"{profession:20s} | Male TPR: {male_tpr:.3f}  Female TPR: {female_tpr:.3f}  Gap: {gap:.3f}")
        print(f"{profession:20s} | TP: {tp:3d}  TN: {tn:4d}  FP: {fp:3d}  FN: {fn:3d}")
        print("-" * 55)

    # Plot
    profession_order = [id2profession[i] for i in range(len(id2profession))]
    gaps_ordered = [profession_tpr_gap.get(p, 0.0) for p in profession_order]

    import os
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"tpr_gap_lambda{lambda_val:.2f}.png")

    plt.figure(figsize=(12, 6))
    sns.barplot(x=profession_order, y=gaps_ordered, color="#6D247A")
    plt.xticks(rotation=90)
    plt.ylabel("TPR Gap (|male − female|)")
    plt.title(f"TPR Gap by Profession (λ = {lambda_val:.2f})")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return {
        "tpr_male": tpr_male,
        "tpr_female": tpr_female,
        "tpr_gap": tpr_gap,
        "tpr_gap_prof": profession_tpr_gap,
        "confusion_stats": confusion_stats,
    }
