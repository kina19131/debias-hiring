"""
Lambda sweep: train one model per lambda value and collect fairness + accuracy results.

Usage:
    python experiments/run_sweep.py [--epochs 15] [--wandb-project debias-hiring]
"""

import argparse
import gc
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from data.dataset import load_bios
from models.classifier import GRUClassifier
from models.distilbert_classifier import DistilBertClassifier
from models.adversary import Adversary, VanillaAdversary
from models.embeddings import load_glove
from training.train import train, set_seed


LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def run_experiment(
    lambda_val: float,
    train_loader,
    valid_loader,
    id2profession: dict,
    device: torch.device,
    epochs: int,
    warmup_epochs: int,
    lr: float,
    lr_adv: float,
    wandb_project: str,
    seed: int,
    adversary_type: str = "label_conditioned",
    model_type: str = "distilbert",
    # GRU-only options (ignored when model_type="distilbert")
    vocab: dict = None,
    glove_path: str = "",
) -> dict:
    print(f"\n--- λ = {lambda_val}  adversary = {adversary_type}  backbone = {model_type} ---")
    set_seed(seed)

    num_classes = len(id2profession)

    if model_type == "distilbert":
        model = DistilBertClassifier(num_classes=num_classes).to(device)
        # Adversary uses hidden_dim * 2 internally (designed for BiGRU).
        # DistilBERT [CLS] is 768-dim, so pass 768 // 2 = 384 so that 384*2 = 768.
        adv_hidden_dim = model.hidden_dim // 2
    else:
        glove_weights = load_glove(vocab, glove_path)
        model = GRUClassifier(
            vocab_size=len(vocab),
            embed_dim=100,
            hidden_dim=128,
            num_classes=num_classes,
            pretrained_weights=glove_weights,
        ).to(device)
        adv_hidden_dim = 128  # GRU hidden_dim; adversary does 128*2=256

    if adversary_type == "vanilla":
        adversary = VanillaAdversary(hidden_dim=adv_hidden_dim).to(device)
    else:
        adversary = Adversary(hidden_dim=adv_hidden_dim, num_classes=num_classes).to(device)

    return train(
        model=model,
        adversary=adversary,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        id2profession=id2profession,
        max_lambda=lambda_val,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        lr=lr,
        lr_adv=lr_adv,
        wandb_project=wandb_project,
        seed=seed,
        adversary_type=adversary_type,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-adv", type=float, default=1e-4,
                        help="Adversary learning rate. Lower than --lr to prevent classifier collapse.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=128,
                        help="Token sequence length. 128 is sufficient for DistilBERT.")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of data to use (0.0-1.0). Use 0.02 for sanity runs.")
    parser.add_argument("--glove-path", type=str, default="glove.6B.100d.txt",
                        help="GRU only — ignored when --model-type=distilbert.")
    parser.add_argument("--wandb-project", type=str, default="debias-hiring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adversary-type",
        choices=["label_conditioned", "vanilla"],
        default="label_conditioned",
        help="'label_conditioned' (ours) vs 'vanilla' (Zhang et al. 2018 baseline)",
    )
    parser.add_argument(
        "--model-type",
        choices=["distilbert", "gru"],
        default="distilbert",
        help="Backbone: 'distilbert' (default) or 'gru' (lightweight ablation)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"sweep_{args.adversary_type}.log"),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}  backbone: {args.model_type}  adversary: {args.adversary_type}")

    train_loader, valid_loader, _, vocab_or_tok, id2profession = load_bios(
        batch_size=args.batch_size,
        max_len=args.max_len,
        subset=args.subset,
        model_type=args.model_type,
    )

    results = {}
    for lam in LAMBDA_VALUES:
        r = run_experiment(
            lambda_val=lam,
            train_loader=train_loader,
            valid_loader=valid_loader,
            id2profession=id2profession,
            device=device,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            lr=args.lr,
            lr_adv=args.lr_adv,
            wandb_project=args.wandb_project,
            seed=args.seed,
            adversary_type=args.adversary_type,
            model_type=args.model_type,
            vocab=vocab_or_tok if args.model_type == "gru" else None,
            glove_path=args.glove_path,
        )
        # Store only metrics — drop model objects so GPU memory is freed
        results[lam] = {k: v for k, v in r.items() if k not in ("model", "adversary")}
        del r
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"λ={lam} done. GPU cache cleared.")

    logging.info(f"\n=== Sweep Summary ({args.adversary_type}) ===")
    logging.info(f"{'lambda':>8}  {'val_acc':>8}  {'tpr_gap':>8}  {'odd_gap':>8}")
    for lam, r in results.items():
        logging.info(
            f"{lam:>8.2f}  {r['val_clf_accuracy']:>8.4f}  "
            f"{r['median_tpr_gap']:>8.4f}  {r['median_odd_gap']:>8.4f}"
        )


if __name__ == "__main__":
    main()
