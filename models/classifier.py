"""
Bidirectional GRU classifier for occupation prediction.
"""

from typing import Optional

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        pretrained_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor):
        """
        Returns:
            logits: (batch, num_classes)
            h:      (batch, hidden_dim * 2)  — used by the adversary
        """
        x = self.embedding(input_ids)
        _, h_n = self.gru(x)
        # Concatenate last hidden states from both directions (1-layer BiGRU)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h = self.norm(h)
        h = self.dropout(h)
        return self.fc(h), h
