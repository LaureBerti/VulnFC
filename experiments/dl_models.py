"""
dl_models.py — PyTorch LSTM / Bi-LSTM / RNN classifiers for VulnFC experiments.
Hyperparameters match Appendix A (Table A.I).
Shared by eval_attack_comparison.py (C9) and eval_structured.py (C8).

Last updated: 2026-04-16
"""

import pickle
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



class Vocab:
    PAD = 0
    UNK = 1

    def __init__(self, min_freq: int = 2, max_size: int = 50_000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.w2i: dict = {"<PAD>": 0, "<UNK>": 1}
        self.i2w: list = ["<PAD>", "<UNK>"]

    def build(self, texts: list) -> None:
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        for word, cnt in counts.most_common(self.max_size - 2):
            if cnt < self.min_freq:
                break
            if word not in self.w2i:
                self.w2i[word] = len(self.i2w)
                self.i2w.append(word)

    def encode(self, text: str, max_len: int) -> list:
        tokens = text.lower().split()[:max_len]
        ids    = [self.w2i.get(t, self.UNK) for t in tokens]
        ids   += [self.PAD] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.i2w)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"w2i": self.w2i, "i2w": self.i2w,
                 "min_freq": self.min_freq, "max_size": self.max_size},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "rb") as f:
            data = pickle.load(f)
        v          = cls(data.get("min_freq", 2), data.get("max_size", 50_000))
        v.w2i      = data["w2i"]
        v.i2w      = data["i2w"]
        return v



class ClaimDataset(Dataset):
    def __init__(self, records: list, vocab: Vocab, max_len: int):
        self.data    = records
        self.vocab   = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        ids = self.vocab.encode(rec["claim"], self.max_len)
        return (
            torch.tensor(ids,          dtype=torch.long),
            torch.tensor(rec["label"], dtype=torch.long),
        )



class LSTMClassifier(nn.Module):
    """LSTM / Bi-LSTM / plain-RNN classifier (Appendix A, Table A.I)."""

    def __init__(
        self,
        vocab_size:    int,
        emb_dim:       int   = 100,
        hidden_dim:    int   = 128,
        n_layers:      int   = 2,
        dropout:       float = 0.30,
        bidirectional: bool  = False,
        rnn_type:      str   = "lstm",
        n_classes:     int   = 2,
    ):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, emb_dim, padding_idx=Vocab.PAD)
        rnn_cls         = nn.LSTM if rnn_type == "lstm" else nn.RNN
        self.rnn        = rnn_cls(
            emb_dim, hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout       = nn.Dropout(dropout)
        factor             = 2 if bidirectional else 1
        self.fc            = nn.Linear(hidden_dim * factor, n_classes)
        self._rnn_type     = rnn_type
        self._bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        if self._rnn_type == "lstm":
            _, (hn, _) = self.rnn(emb)
        else:
            _, hn = self.rnn(emb)
        if self._bidirectional:
            hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            hidden = hn[-1]
        return self.fc(self.dropout(hidden))



MODEL_CONFIGS = {
    "lstm_title": {
        "rnn_type": "lstm", "bidirectional": False, "n_layers": 2,
        "hidden_dim": 128,  "emb_dim": 100, "dropout": 0.30, "max_len": 20,
    },
    "lstm_text": {
        "rnn_type": "lstm", "bidirectional": False, "n_layers": 2,
        "hidden_dim": 128,  "emb_dim": 100, "dropout": 0.30, "max_len": 1000,
    },
    "bi_lstm": {
        "rnn_type": "lstm", "bidirectional": True, "n_layers": 2,
        "hidden_dim": 128,  "emb_dim": 100, "dropout": 0.30, "max_len": 20,
    },
    "rnn": {
        "rnn_type": "rnn",  "bidirectional": False, "n_layers": 1,
        "hidden_dim": 128,  "emb_dim": 100, "dropout": 0.20, "max_len": 20,
    },
}


def build_model(model_type: str, vocab_size: int) -> LSTMClassifier:
    cfg = {k: v for k, v in MODEL_CONFIGS[model_type].items() if k != "max_len"}
    return LSTMClassifier(vocab_size=vocab_size, **cfg)



def train_model(
    model_type:     str,
    train_data:     list,
    val_data:       list,
    checkpoint_dir: str,
    device:         str = "cpu",
    seed:           int = 42,
) -> tuple:
    """Train a DL model from scratch and save checkpoint. Returns (model, vocab)."""
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        raise ImportError("pip install scikit-learn")

    torch.manual_seed(seed)
    random.seed(seed)

    cfg     = MODEL_CONFIGS[model_type]
    max_len = cfg["max_len"]

    vocab = Vocab()
    vocab.build([r["claim"] for r in train_data])

    train_ds = ClaimDataset(train_data, vocab, max_len)
    val_ds   = ClaimDataset(val_data,   vocab, max_len)
    tr_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    va_loader = DataLoader(val_ds,   batch_size=64, shuffle=False)

    model     = build_model(model_type, len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    patience    = 3
    wait        = 0
    best_state  = None
    t0          = time.time()

    for epoch in range(30):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for X, y in va_loader:
                preds.extend(model(X.to(device)).argmax(1).cpu().tolist())
                gts.extend(y.tolist())

        val_f1 = f1_score(gts, preds, average="binary", zero_division=0)
        print(f"  epoch {epoch+1:2d}/30  val_f1={val_f1:.4f}  elapsed={time.time()-t0:.0f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping — best_val_f1={best_val_f1:.4f}")
                break

    if best_state:
        model.load_state_dict(best_state)

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path / "model.pt")
    vocab.save(str(ckpt_path / "vocab.pkl"))
    print(f"  [train] Checkpoint → {ckpt_path}  best_val_f1={best_val_f1:.4f}")
    return model, vocab



def load_model(model_type: str, checkpoint_dir: str, device: str = "cpu") -> tuple:
    ckpt_path = Path(checkpoint_dir)
    vocab     = Vocab.load(str(ckpt_path / "vocab.pkl"))
    model     = build_model(model_type, len(vocab)).to(device)
    state     = torch.load(ckpt_path / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, vocab



def predict_batch(
    model:      LSTMClassifier,
    records:    list,
    vocab:      Vocab,
    max_len:    int,
    device:     str = "cpu",
    batch_size: int = 64,
) -> list:
    """Return predicted class labels for a list of {claim, label} records."""
    ds     = ClaimDataset(records, vocab, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds  = []
    model.eval()
    with torch.no_grad():
        for X, _ in loader:
            preds.extend(model(X.to(device)).argmax(1).cpu().tolist())
    return preds
