"""
DistilBERT classifier for occupation prediction.

Drop-in replacement for GRUClassifier — same (logits, h) output contract.
hidden_dim = 768 (DistilBERT hidden size).
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertClassifier(nn.Module):
    """
    DistilBERT backbone + linear head.
    Returns the [CLS] token as h, which is passed to the adversary unchanged.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.hidden_dim = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)  1 = real token, 0 = padding

        Returns:
            logits: (batch, num_classes)
            h:      (batch, 768) — [CLS] representation used by the adversary
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state[:, 0, :])  # [CLS] token
        return self.fc(h), h
