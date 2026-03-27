"""
Two debiasing baselines for comparison against adversarial training.

1. Token masking (scrubbing)
   Replace a fixed list of explicitly gendered tokens with <unk> at inference
   time, before the representations are computed. This is the simplest possible
   debiasing intervention — no model changes, no training.

2. INLP — Iterative Nullspace Projection (Ravfogel et al., ACL 2020)
   Starting from a frozen encoder, iteratively find the linear direction that
   best predicts gender from the representations and project it out. Repeat
   until a linear probe can no longer predict gender above chance. The resulting
   projection matrix is then applied at test time.

Both baselines are evaluated with the same fairness metrics as adversarial
training (per-profession TPR gap, equalized odds gap).

Usage
-----
    from evaluation.baselines import evaluate_token_masking, evaluate_inlp

    masking_results = evaluate_token_masking(model, test_loader, device, id2profession, vocab)
    inlp_results    = evaluate_inlp(model, train_loader, test_loader, device, id2profession)
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from evaluation.metrics import compute_equalized_opps, compute_equalized_odds


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _collect_predictions(model, dataloader, device, input_ids_transform=None):
    """
    Run model forward pass over dataloader, optionally transforming input_ids.

    Returns tensors: y_true, y_pred, genders, and all hidden states H.
    """
    model.eval()
    y_trues, y_preds, genders_all, hs = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            if input_ids_transform is not None:
                input_ids = input_ids_transform(input_ids)

            labels = batch["label"].to(device)
            gender = batch["gender"].to(device)

            logits, h = model(input_ids)
            y_trues.append(labels.cpu())
            y_preds.append(logits.argmax(1).cpu())
            genders_all.append(gender.cpu())
            hs.append(h.cpu())

    return (
        torch.cat(y_trues),
        torch.cat(y_preds),
        torch.cat(genders_all),
        torch.cat(hs),
    )


def _fairness_summary(y_true, y_pred, genders, id2profession, tag: str) -> dict:
    eopp = compute_equalized_opps(y_true, y_pred, genders, id2profession, epoch=0)
    eodd = compute_equalized_odds(y_true, y_pred, genders, id2profession, epoch=0)

    median_tpr_gap = np.median([
        abs(s["TPR_female"] - s["TPR_male"])
        for s in eopp
        if not (np.isnan(s["TPR_female"]) or np.isnan(s["TPR_male"]))
    ])
    median_odd_gap = np.median([s["Odd_gap"] for s in eodd if not np.isnan(s["Odd_gap"])])
    accuracy = (y_true == y_pred).float().mean().item()

    print(f"\n[{tag}] accuracy={accuracy:.4f}  median TPR gap={median_tpr_gap:.4f}  "
          f"median Odd gap={median_odd_gap:.4f}")

    return {
        "accuracy": accuracy,
        "median_tpr_gap": median_tpr_gap,
        "median_odd_gap": median_odd_gap,
        "eopp_stats": eopp,
        "eodd_stats": eodd,
    }


# ---------------------------------------------------------------------------
# Baseline 1: Token masking
# ---------------------------------------------------------------------------

# Explicitly gendered surface forms. Sourced from common gender-debiasing
# word lists (Bolukbasi et al. 2016; Zhao et al. 2018).
GENDERED_TOKENS = frozenset([
    # pronouns
    "he", "she", "him", "her", "his", "hers", "himself", "herself",
    # titles
    "mr", "mrs", "ms", "miss", "sir", "madam",
    # kinship
    "man", "woman", "boy", "girl", "male", "female",
    "father", "mother", "dad", "mom", "son", "daughter",
    "brother", "sister", "husband", "wife",
    "grandfather", "grandmother", "grandson", "granddaughter",
    "uncle", "aunt", "nephew", "niece",
    # compound kin
    "boyfriend", "girlfriend", "fiance", "fiancee",
    # occupational gender markers
    "actor", "actress", "waiter", "waitress",
    "steward", "stewardess",
])


def _build_gendered_id_set(vocab: dict[str, int]) -> set[int]:
    return {vocab[w] for w in GENDERED_TOKENS if w in vocab}


def make_masking_transform(vocab: dict[str, int]):
    """
    Returns a function that replaces gendered token indices with <unk> (idx 1)
    in a batch of input_ids tensors.
    """
    gendered_ids = _build_gendered_id_set(vocab)
    unk_id = vocab.get("<unk>", 1)

    def transform(input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for gid in gendered_ids:
            mask |= input_ids == gid
        return input_ids.masked_fill(mask, unk_id)

    return transform


def evaluate_token_masking(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    id2profession: dict,
    vocab: dict[str, int],
) -> dict:
    """
    Evaluate fairness after replacing gendered tokens with <unk>.
    No model retraining — masking happens at inference time.
    """
    transform = make_masking_transform(vocab)
    y_true, y_pred, genders, _ = _collect_predictions(
        model, test_loader, device, input_ids_transform=transform
    )
    return _fairness_summary(y_true, y_pred, genders, id2profession, "token_masking")


# ---------------------------------------------------------------------------
# Baseline 2: INLP (Ravfogel et al. 2020)
# ---------------------------------------------------------------------------

def _rowspace_projection(W: np.ndarray) -> np.ndarray:
    """Projection matrix onto the rowspace of W."""
    if W.ndim == 1:
        W = W.reshape(1, -1)
    rank = np.linalg.matrix_rank(W)
    Q, _ = np.linalg.qr(W.T)
    Q = Q[:, :rank]
    return Q @ Q.T


def fit_inlp(
    H: np.ndarray,
    gender_labels: np.ndarray,
    n_iterations: int = 30,
    min_acc_above_chance: float = 0.01,
) -> np.ndarray:
    """
    Fit an INLP projection matrix that removes the linear gender subspace from H.

    Algorithm:
        1. Train logistic regression to predict gender from H.
        2. Project H onto the nullspace of the classifier weight vector.
        3. Repeat until the probe's accuracy is at most chance + min_acc_above_chance.

    Args:
        H:                    (n_samples, hidden_dim) representation matrix.
        gender_labels:        (n_samples,) binary gender labels (0/1).
        n_iterations:         maximum number of projection rounds.
        min_acc_above_chance: stop when probe acc ≤ 0.5 + this value.

    Returns:
        P: (hidden_dim, hidden_dim) projection matrix.
           Apply as H_debiased = H @ P.T
    """
    dim = H.shape[1]
    P_cumulative = np.eye(dim)
    H_projected = H.copy()
    chance = gender_labels.mean()
    chance = max(chance, 1 - chance)  # majority-class baseline

    for i in range(n_iterations):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(H_projected, gender_labels)
        acc = clf.score(H_projected, gender_labels)

        print(f"  INLP iter {i+1:2d}: linear probe acc = {acc:.4f}  (chance = {chance:.4f})")

        if acc <= chance + min_acc_above_chance:
            print(f"  Converged at iteration {i+1}.")
            break

        W = clf.coef_  # (1, hidden_dim)
        P_i = np.eye(dim) - _rowspace_projection(W)
        H_projected = H_projected @ P_i.T
        P_cumulative = P_i @ P_cumulative

    return P_cumulative


def evaluate_inlp(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    id2profession: dict,
    n_iterations: int = 30,
) -> dict:
    """
    1. Extract representations from train split.
    2. Fit INLP projection matrix P on those representations.
    3. Project test representations with P.
    4. Retrain a linear classification head on projected train reps.
    5. Evaluate fairness on test set with the projected + re-classified outputs.

    Note: step 4 is necessary because the original FC head was trained on
    unmodified representations; applying P changes the geometry.
    """
    print("\nExtracting train representations for INLP...")
    y_true_tr, _, genders_tr, H_tr = _collect_predictions(model, train_loader, device)
    print(f"  Train reps: {H_tr.shape}")

    print("Fitting INLP projection matrix...")
    P = fit_inlp(H_tr.numpy(), genders_tr.numpy(), n_iterations=n_iterations)
    P_t = torch.tensor(P, dtype=torch.float32).to(device)

    # Retrain a linear classification head on projected train representations
    H_tr_proj = H_tr.numpy() @ P.T
    print("Retraining linear head on projected representations...")
    head = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="multinomial", C=1.0)
    head.fit(H_tr_proj, y_true_tr.numpy())
    train_acc = head.score(H_tr_proj, y_true_tr.numpy())
    print(f"  Linear head train accuracy: {train_acc:.4f}")

    # Evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    y_trues, y_preds, genders_te = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            gender = batch["gender"].to(device)

            _, h = model(input_ids)
            h_proj = (h @ P_t.T).cpu().numpy()
            preds = head.predict(h_proj)

            y_trues.append(labels.cpu())
            y_preds.append(torch.tensor(preds, dtype=torch.long))
            genders_te.append(gender.cpu())

    y_true = torch.cat(y_trues)
    y_pred = torch.cat(y_preds)
    genders = torch.cat(genders_te)

    return _fairness_summary(y_true, y_pred, genders, id2profession, "inlp")
