"""
Run all debiasing methods and print Table 1 (the paper's main comparison).

Requires:
  - A trained baseline model checkpoint (lambda=0, label_conditioned run)
  - A trained label-conditioned adversary checkpoint (best lambda)
  - A trained vanilla adversary checkpoint (best lambda)

Usage:
    python experiments/compare_baselines.py \
        --baseline-ckpt baseline_checkpoint.pth \
        --lc-ckpt adv_1.00_checkpoint.pth \
        --vanilla-ckpt vanilla_adv_1.00_checkpoint.pth \
        --glove-path glove.6B.100d.txt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from data.dataset import load_bios
from models.classifier import GRUClassifier
from models.adversary import Adversary, VanillaAdversary
from models.embeddings import load_glove
from evaluation.baselines import evaluate_token_masking, evaluate_inlp
from evaluation.metrics import compute_equalized_opps, compute_equalized_odds
import numpy as np


def load_gru(ckpt_path, vocab, id2profession, device):
    model = GRUClassifier(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=128,
        num_classes=len(id2profession),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def eval_classifier(model, test_loader, device, id2profession, tag):
    y_trues, y_preds, genders = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            gender = batch["gender"].to(device)
            logits, _ = model(input_ids)
            y_trues.append(labels.cpu())
            y_preds.append(logits.argmax(1).cpu())
            genders.append(gender.cpu())

    y_true = torch.cat(y_trues)
    y_pred = torch.cat(y_preds)
    genders = torch.cat(genders)

    eopp = compute_equalized_opps(y_true, y_pred, genders, id2profession, epoch=0)
    eodd = compute_equalized_odds(y_true, y_pred, genders, id2profession, epoch=0)

    acc = (y_true == y_pred).float().mean().item()
    tpr_gap = np.median([
        abs(s["TPR_female"] - s["TPR_male"])
        for s in eopp if not (np.isnan(s["TPR_female"]) or np.isnan(s["TPR_male"]))
    ])
    odd_gap = np.median([s["Odd_gap"] for s in eodd if not np.isnan(s["Odd_gap"])])
    return {"tag": tag, "accuracy": acc, "median_tpr_gap": tpr_gap, "median_odd_gap": odd_gap}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ckpt", required=True, help="λ=0 checkpoint path")
    parser.add_argument("--lc-ckpt", required=True, help="label-conditioned adversary checkpoint")
    parser.add_argument("--vanilla-ckpt", required=True, help="vanilla adversary checkpoint")
    parser.add_argument("--glove-path", default="glove.6B.100d.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inlp-iters", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader, vocab, id2profession = load_bios(
        batch_size=args.batch_size
    )

    rows = []

    # --- No debiasing ---
    model_base = load_gru(args.baseline_ckpt, vocab, id2profession, device)
    rows.append(eval_classifier(model_base, test_loader, device, id2profession, "No debiasing"))

    # --- Token masking ---
    r = evaluate_token_masking(model_base, test_loader, device, id2profession, vocab)
    rows.append({"tag": "Token masking", **{k: r[k] for k in ("accuracy", "median_tpr_gap", "median_odd_gap")}})

    # --- INLP ---
    r = evaluate_inlp(model_base, train_loader, test_loader, device, id2profession, n_iterations=args.inlp_iters)
    rows.append({"tag": "INLP", **{k: r[k] for k in ("accuracy", "median_tpr_gap", "median_odd_gap")}})

    # --- Vanilla adversary ---
    model_vanilla = load_gru(args.vanilla_ckpt, vocab, id2profession, device)
    rows.append(eval_classifier(model_vanilla, test_loader, device, id2profession, "Vanilla adversary"))

    # --- Label-conditioned adversary (ours) ---
    model_lc = load_gru(args.lc_ckpt, vocab, id2profession, device)
    rows.append(eval_classifier(model_lc, test_loader, device, id2profession, "Label-conditioned (ours)"))

    # --- Print Table 1 ---
    print("\n" + "=" * 65)
    print(f"{'Method':<30}  {'Accuracy':>8}  {'TPR Gap↓':>8}  {'Odd Gap↓':>8}")
    print("-" * 65)
    for r in rows:
        marker = " *" if r["tag"] == "Label-conditioned (ours)" else ""
        print(f"{r['tag']:<30}  {r['accuracy']:>8.4f}  {r['median_tpr_gap']:>8.4f}  {r['median_odd_gap']:>8.4f}{marker}")
    print("=" * 65)
    print("* = proposed method")


if __name__ == "__main__":
    main()
