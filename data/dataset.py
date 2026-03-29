"""
Bias in Bios dataset loading, tokenization, and PyTorch Dataset wrapper.

Supports two backends:
  model_type="distilbert"  — uses HuggingFace DistilBertTokenizerFast; returns
                             input_ids + attention_mask in every batch (default)
  model_type="gru"         — uses the original custom vocab + GloVe pipeline
"""

import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def basic_tokenizer(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(dataset, tokenizer=basic_tokenizer, min_freq: int = 2) -> dict[str, int]:
    """Build vocabulary from a HuggingFace dataset split. Pass the train split only."""
    counter = Counter()
    for text in dataset["hard_text"]:
        counter.update(tokenizer(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1
    return vocab


def tokens_to_ids(tokens: list[str], vocab: dict[str, int], max_len: int = 256) -> list[int]:
    unk_id = vocab.get("<unk>", 0)
    pad_id = vocab.get("<pad>", 0)
    ids = [vocab.get(t, unk_id) for t in tokens]
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class BiosDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, texts, genders=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.texts = texts
        self.genders = genders

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": self.texts[idx],
        }
        if self.genders is not None:
            item["gender"] = self.genders[idx]
        return item


def process_split(raw_dataset, vocab: dict[str, int], max_len: int = 256) -> BiosDataset:
    """GRU path: custom vocab tokenization."""
    texts = list(raw_dataset["hard_text"])
    tokenized = [basic_tokenizer(t) for t in texts]
    input_ids = [tokens_to_ids(t, vocab, max_len) for t in tokenized]
    attention_mask = [[1 if x != 0 else 0 for x in ids] for ids in input_ids]
    labels = list(raw_dataset["profession"])
    genders = list(raw_dataset["gender"]) if "gender" in raw_dataset.column_names else None
    return BiosDataset(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, texts=texts, genders=genders,
    )


def process_split_bert(raw_dataset, tokenizer, max_len: int = 128) -> BiosDataset:
    """DistilBERT path: HuggingFace tokenizer."""
    texts = list(raw_dataset["hard_text"])
    enc = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = list(raw_dataset["profession"])
    genders = list(raw_dataset["gender"]) if "gender" in raw_dataset.column_names else None
    return BiosDataset(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, texts=texts, genders=genders,
    )


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

ID2PROFESSION = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor", 4: "comedian",
    5: "composer", 6: "dentist", 7: "dietitian", 8: "dj", 9: "filmmaker",
    10: "interior_designer", 11: "journalist", 12: "model", 13: "nurse", 14: "painter",
    15: "paralegal", 16: "pastor", 17: "personal_trainer", 18: "photographer", 19: "physician",
    20: "poet", 21: "professor", 22: "psychologist", 23: "rapper", 24: "software_engineer",
    25: "surgeon", 26: "teacher", 27: "yoga_teacher",
}


def load_bios(
    batch_size: int = 32,
    max_len: int = 128,
    min_freq: int = 2,
    subset: float = 1.0,
    model_type: str = "distilbert",
):
    """
    Load Bias in Bios from HuggingFace and return
    (train_loader, valid_loader, test_loader, vocab_or_tokenizer, id2profession).

    Args:
        model_type: "distilbert" (default) or "gru"
        max_len:    sequence length. 128 is sufficient for DistilBERT on bios.
        subset:     fraction of data to use (0.0–1.0). Use 0.02 for sanity runs.
    """
    train_raw = load_dataset("LabHC/bias_in_bios", split="train")
    valid_raw = load_dataset("LabHC/bias_in_bios", split="dev")
    test_raw  = load_dataset("LabHC/bias_in_bios", split="test")

    if subset < 1.0:
        train_raw = train_raw.select(range(int(len(train_raw) * subset)))
        valid_raw = valid_raw.select(range(int(len(valid_raw) * subset)))
        test_raw  = test_raw.select(range(int(len(test_raw)  * subset)))

    if model_type == "distilbert":
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        train_ds = process_split_bert(train_raw, tokenizer, max_len)
        valid_ds = process_split_bert(valid_raw, tokenizer, max_len)
        test_ds  = process_split_bert(test_raw,  tokenizer, max_len)
        vocab_or_tok = tokenizer
    else:
        vocab = build_vocab(train_raw, min_freq=min_freq)
        train_ds = process_split(train_raw, vocab, max_len)
        valid_ds = process_split(valid_raw, vocab, max_len)
        test_ds  = process_split(test_raw,  vocab, max_len)
        vocab_or_tok = vocab

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader, test_loader, vocab_or_tok, ID2PROFESSION
