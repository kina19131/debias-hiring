"""
Loss functions for the classifier and adversary.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight


def build_clf_criterion(train_loader, device: torch.device) -> nn.CrossEntropyLoss:
    """Weighted cross-entropy loss; weights are inversely proportional to class frequency."""
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch["label"].tolist())
    labels = np.array(all_labels)
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))


def build_adv_criterion() -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss()
