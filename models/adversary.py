"""
Label-conditioned adversary with gradient reversal.

The key architectural novelty: instead of feeding the hidden state h directly
to the adversary, we construct:

    adv_input = [h,  h * y,  h * (1 - y)]

where y is the one-hot occupation label (broadcast as a scalar per sample).
This forces the model to remove gender information *conditional on occupation*,
which is the correct inductive bias for the hiring domain — a surgeon's bio
should not leak gender regardless of what occupation features are active.

Reference: extends Zhang et al. (2018) "Mitigating Unwanted Biases with
Adversarial Learning" with occupation-conditional gradient reversal.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_val: float) -> torch.Tensor:
    return GradReverse.apply(x, lambda_val)


# ---------------------------------------------------------------------------
# Adversary
# ---------------------------------------------------------------------------

class VanillaAdversary(nn.Module):
    """
    Baseline adversary (Zhang et al. 2018): predicts gender directly from h.
    Input dimension: hidden_dim * 2  (= 256 for hidden_dim=128).
    Used as the comparison point for Adversary to isolate the effect of
    label-conditioning.
    """

    def __init__(self, hidden_dim: int, adv_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, adv_hidden),
            nn.ReLU(),
            nn.Linear(adv_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, h: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        # true_labels ignored — vanilla adversary sees only h
        return self.net(h).squeeze(1)


class Adversary(nn.Module):
    """
    Label-conditioned adversary (ours): predicts gender from occupation-conditioned
    representation.

    Input dimension: hidden_dim * 2 * 3  (= 768 for hidden_dim=128)
    because adv_input = cat([h, h*label, h*(1-label)]).
    """

    def __init__(self, hidden_dim: int, adv_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 3, adv_hidden),
            nn.ReLU(),
            nn.Linear(adv_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, h: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:           (batch, hidden_dim * 2) — hidden state from GRUClassifier
            true_labels: (batch,) or (batch, 1) — integer occupation class ids

        Returns:
            logits: (batch,) — raw (pre-sigmoid) gender prediction
        """
        if true_labels.dim() == 1:
            true_labels = true_labels.unsqueeze(1)
        true_labels = true_labels.float().to(h.device)

        h1 = h * true_labels          # occupation-present component
        h0 = h * (1.0 - true_labels)  # occupation-absent component
        adv_input = torch.cat([h, h1, h0], dim=1)
        return self.net(adv_input).squeeze(1)
