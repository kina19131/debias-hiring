"""
Main training loop: GRUClassifier + label-conditioned adversary with gradient reversal.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from models.adversary import grad_reverse
from evaluation.metrics import compute_equalized_opps, compute_equalized_odds
from training.losses import build_clf_criterion, build_adv_criterion


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits, _ = model(input_ids)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            del input_ids, labels, logits
    return correct / total


def evaluate_loss(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(input_ids)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            total += labels.size(0)
    return total_loss / total


def evaluate_adversary(model, adversary, dataloader, device) -> float:
    model.eval()
    adversary.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            genders = batch["gender"].float().to(device)
            labels = batch["label"].to(device)
            _, h = model(input_ids)
            preds = (adversary(h, labels) > 0).float()
            correct += (preds == genders).sum().item()
            total += genders.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _build_tag(lambda_adv: float) -> str:
    return "baseline" if lambda_adv == 0 else f"adv_{lambda_adv:.2f}"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    adversary: nn.Module,
    train_loader,
    valid_loader,
    device: torch.device,
    id2profession: dict,
    max_lambda: float = 0.0,
    epochs: int = 10,
    warmup_epochs: int = 5,
    lr: float = 1e-3,
    wandb_project: str = "debias-hiring",
    seed: int = 42,
) -> dict:
    """
    Train GRUClassifier with the label-conditioned adversary.

    Lambda is linearly warmed up from 0 to max_lambda over warmup_epochs,
    then held constant. When max_lambda == 0 the adversary is not updated
    (pure classification baseline).

    Best checkpoint is selected by median TPR gap (lower is better).
    """
    set_seed(seed)

    tag = _build_tag(max_lambda)
    run = wandb.init(
        project=wandb_project,
        config=dict(
            epochs=epochs,
            batch_size=train_loader.batch_size,
            embed_dim=100,
            hidden_dim=128,
            lambda_adv=max_lambda,
            warmup_epochs=warmup_epochs,
            lr=lr,
            seed=seed,
        ),
        reinit=True,
    )

    clf_crit = build_clf_criterion(train_loader, device)
    adv_crit = build_adv_criterion()

    optim_clf = torch.optim.Adam(model.parameters(), lr=lr)
    optim_adv = torch.optim.Adam(adversary.parameters(), lr=lr)

    best_eo_gap = float("inf")
    best_state = None

    for epoch in range(epochs):
        lambda_val = min(max_lambda, (epoch / max(warmup_epochs, 1)) * max_lambda)
        model.train()
        adversary.train()

        tot_clf_loss = tot_adv_loss = 0.0
        tot_clf_ok = tot_adv_ok = tot = 0

        for batch in tqdm(train_loader, desc=f"[{tag}] epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            y_clf = batch["label"].to(device)
            y_adv = batch["gender"].float().to(device)

            optim_clf.zero_grad()
            optim_adv.zero_grad()

            logits, h = model(input_ids)
            clf_loss = clf_crit(logits, y_clf)

            if lambda_val == 0:
                with torch.no_grad():
                    adversary(h.detach(), y_clf)
                adv_loss = torch.tensor(0.0, device=device)
                total_loss = clf_loss
            else:
                rev_h = grad_reverse(h, lambda_val)
                gender_logits = adversary(rev_h, y_clf)
                adv_loss = adv_crit(gender_logits, y_adv)
                total_loss = clf_loss + adv_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim_clf.step()

            if lambda_val > 0:
                torch.nn.utils.clip_grad_norm_(adversary.parameters(), max_norm=5.0)
                optim_adv.step()

            tot += y_clf.size(0)
            tot_clf_loss += clf_loss.item()
            tot_clf_ok += (logits.argmax(1) == y_clf).sum().item()
            if lambda_val > 0:
                tot_adv_loss += adv_loss.item()
                adv_preds = (gender_logits > 0).float()
                tot_adv_ok += (adv_preds == y_adv).sum().item()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        val_acc = evaluate(model, valid_loader, device)
        val_clf_loss = evaluate_loss(model, valid_loader, clf_crit, device)
        val_adv_acc = evaluate_adversary(model, adversary, valid_loader, device)

        # Collect predictions for fairness metrics
        y_true_all, y_pred_all, gender_all = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                y_clf = batch["label"].to(device)
                y_gender = batch["gender"].to(device)
                logits, _ = model(input_ids)
                y_true_all.append(y_clf)
                y_pred_all.append(logits.argmax(1))
                gender_all.append(y_gender)

        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        gender_all = torch.cat(gender_all)

        eopp_stats = compute_equalized_opps(y_true_all, y_pred_all, gender_all, id2profession, epoch + 1)
        eodd_stats = compute_equalized_odds(y_true_all, y_pred_all, gender_all, id2profession, epoch + 1)

        median_tpr_gap = np.median([
            abs(s["TPR_female"] - s["TPR_male"])
            for s in eopp_stats
            if not np.isnan(s["TPR_female"]) and not np.isnan(s["TPR_male"])
        ])
        median_odd_gap = np.median([
            s["Odd_gap"] for s in eodd_stats if not np.isnan(s["Odd_gap"])
        ])

        # Per-profession W&B logs
        prof_logs = {}
        for s in eopp_stats:
            prof_logs[f"prof_{s['profession']}_TPR_diff"] = s["TPR_diff"]
        for s in eodd_stats:
            prof_logs[f"prof_{s['profession']}_Odd_gap"] = s["Odd_gap"]

        run.log({
            "epoch": epoch + 1,
            "lambda_adv": lambda_val,
            "train_clf_loss": tot_clf_loss / len(train_loader),
            "train_adv_loss": tot_adv_loss / max(1, len(train_loader)),
            "train_clf_accuracy": tot_clf_ok / tot,
            "train_adv_accuracy": (tot_adv_ok / tot) if lambda_val > 0 else 0.0,
            "val_clf_accuracy": val_acc,
            "val_adv_accuracy": val_adv_acc,
            "val_clf_loss": val_clf_loss,
            "median_opp_gap": median_tpr_gap,
            "median_odds_gap": median_odd_gap,
            **prof_logs,
        })

        # Track best model by TPR gap
        if median_tpr_gap < best_eo_gap:
            best_eo_gap = median_tpr_gap
            best_state = {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "adversary": {k: v.cpu() for k, v in adversary.state_dict().items()},
                "epoch": epoch,
                "lambda_val": lambda_val,
                "eo_gap": best_eo_gap,
            }

    # ------------------------------------------------------------------
    # Save and log best checkpoint
    # ------------------------------------------------------------------
    model.load_state_dict({k: v.to(device) for k, v in best_state["model"].items()})
    adversary.load_state_dict({k: v.to(device) for k, v in best_state["adversary"].items()})

    ckpt_path = f"{tag}_checkpoint.pth"
    torch.save(best_state, ckpt_path)

    art_name = f"joint_model_{tag}"
    artifact = wandb.Artifact(
        name=art_name,
        type="model",
        description=f"GRU + adversary trained with λ={max_lambda}",
        metadata=dict(
            lambda_adv=max_lambda,
            val_clf_accuracy=val_acc,
            val_adv_accuracy=val_adv_acc,
            equalized_opportunity_gap=best_eo_gap,
            epoch=best_state["epoch"],
        ),
    )
    artifact.add_file(ckpt_path)
    run.log_artifact(artifact, aliases=["latest", tag])
    os.remove(ckpt_path)
    run.finish()

    return {
        "val_clf_accuracy": val_acc,
        "val_adv_accuracy": val_adv_acc,
        "median_tpr_gap": median_tpr_gap,
        "median_odd_gap": median_odd_gap,
        "best_eo_gap": best_eo_gap,
        "model": model,
        "adversary": adversary,
    }
