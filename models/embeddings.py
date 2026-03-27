"""
GloVe embedding loader.
"""

import numpy as np
import torch


def load_glove(vocab: dict[str, int], glove_path: str = "glove.6B.100d.txt") -> torch.Tensor:
    """
    Load GloVe vectors for words in vocab.
    Words not found in GloVe are initialised with small uniform random values.

    Args:
        vocab: token → index mapping built from the training corpus.
        glove_path: path to the GloVe text file (e.g. glove.6B.100d.txt).

    Returns:
        Float tensor of shape (vocab_size, embedding_dim).
    """
    embedding_dim = 100
    embedding_matrix = np.random.uniform(-0.05, 0.05, (len(vocab), embedding_dim)).astype(np.float32)

    try:
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                word = tokens[0]
                if word in vocab:
                    embedding_matrix[vocab[word]] = np.array(tokens[1:], dtype=np.float32)
    except FileNotFoundError:
        print(f"Warning: GloVe file '{glove_path}' not found. Using random embeddings.")

    return torch.tensor(embedding_matrix)
