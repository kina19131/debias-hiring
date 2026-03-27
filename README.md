# debias-hiring

Adversarial debiasing for occupation classification on the **Bias in Bios** dataset (De-Arteaga et al., 2019).

## Method

A Bidirectional GRU classifier predicts occupation from biography text. A **label-conditioned adversary** with gradient reversal simultaneously tries to predict gender from the hidden state, forcing the classifier to remove gender information *conditional on occupation class* — the correct inductive bias for the hiring domain.

The adversary input is constructed as:
```
adv_input = cat([h, h * y, h * (1 - y)])   # shape: (batch, 768)
```
where `h` is the BiGRU hidden state and `y` is the occupation label. This extends Zhang et al. (2018) with occupation-conditional gradient reversal.

**Key result:** global TPR gap 0.187 → 0.089 (−52%), accuracy 78% → 77%.

## Repo structure

```
debias-hiring/
├── data/
│   └── dataset.py          # Bias in Bios loading + tokenization + BiosDataset
├── models/
│   ├── classifier.py       # GRUClassifier (BiGRU + LayerNorm + Dropout)
│   ├── adversary.py        # Label-conditioned Adversary + GradReverse
│   └── embeddings.py       # GloVe loader
├── training/
│   ├── train.py            # Main train() loop with warmup lambda + W&B logging
│   └── losses.py           # Weighted CE loss + BCEWithLogits
├── evaluation/
│   ├── metrics.py          # compute_equalized_opps(), compute_equalized_odds(), log_confusion_and_fairness()
│   └── baselines.py        # (TODO) token-masking baseline + INLP
├── experiments/
│   └── run_sweep.py        # Lambda sweep {0, 0.25, 0.5, 0.75, 1.0}
├── configs/
│   └── default.yaml        # All hyperparameters
├── notebooks/
│   └── AI4G_Debias.ipynb   # Original exploratory notebook
├── results/
│   └── figures/
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Download GloVe embeddings (100d)
# https://nlp.stanford.edu/projects/glove/  →  glove.6B.zip

python experiments/run_sweep.py --epochs 15 --glove-path glove.6B.100d.txt
```

## Fairness metrics

- **Equal Opportunity** (`compute_equalized_opps`): per-profession TPR gap between male and female subgroups
- **Equalized Odds** (`compute_equalized_odds`): 0.5 × (TPR gap + FPR gap) per profession
- Both are computed per epoch during training and logged to W&B

## Citation

```
De-Arteaga, M. et al. (2019). Bias in Bios. FAccT.
Zhang, B. H. et al. (2018). Mitigating Unwanted Biases with Adversarial Learning. AIES.
Ravfogel, S. et al. (2020). Null It Out: INLP. ACL.
```
