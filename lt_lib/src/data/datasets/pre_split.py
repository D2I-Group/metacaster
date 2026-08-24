"""Dataset backed by raw pre-windowed .npy files.

Expects two files in ``root_path``:

- ``train.npy`` — ``(B, seq_len + pred_len, C)`` float32, raw (unnormalised)
- ``test.npy``  — ``(B, seq_len + pred_len, C)`` float32, raw

Val is carved from the tail of ``train.npy``.  StandardScaler is fitted on
the train portion and applied to every split.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from benchmark.registry import DATASET_REGISTRY
from data.schemas.datasets.pre_split import PreSplitParameterConfig


class Dataset_PreSplit(Dataset):
    """Pre-windowed ``.npy`` dataset with built-in normalisation.

    Parameters
    ----------
    root_path : str
        Directory containing ``train.npy`` and ``test.npy``.
    data_path : str
        Unused; kept for API compatibility.
    size : tuple[int, int, int]
        ``(seq_len, label_len, pred_len)``.
    flag : str
        ``"train"``, ``"val"``, or ``"test"``.
    val_ratio : float
        Fraction of train windows reserved for validation.
    scale : bool
        Whether to apply StandardScaler.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        val_ratio: float = 0.1,
        scale: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = size

        def _ensure_3d(arr: np.ndarray) -> np.ndarray:
            """Tolerate generators that wrote (B, T) instead of (B, T, 1)
            for univariate datasets — promote silently to a 3-D array.
            """
            if arr.ndim == 2:
                return arr[:, :, None]
            return arr

        # ── Always load train.npy to fit scaler ──────────────────────────
        train_all = _ensure_3d(
            np.load(os.path.join(root_path, "train.npy")).astype(np.float32)
        )
        n_val = max(1, round(len(train_all) * val_ratio))
        n_train = len(train_all) - n_val

        # ── Select windows ───────────────────────────────────────────────
        if flag == "train":
            windows = train_all[:n_train]
        elif flag == "val":
            windows = train_all[n_train:]
        elif flag == "test":
            windows = _ensure_3d(
                np.load(os.path.join(root_path, "test.npy")).astype(np.float32)
            )
        else:
            raise ValueError(f"Invalid flag '{flag}'. Expected train/val/test.")

        # ── Split into x / y ─────────────────────────────────────────────
        self.x = windows[:, : self.seq_len, :]
        self.y = windows[:, self.seq_len - self.label_len :, :]

        # ── Normalisation ────────────────────────────────────────────────
        if scale:
            C = train_all.shape[-1]
            self.scaler = StandardScaler()
            self.scaler.fit(train_all[:n_train].reshape(-1, C))

            self.x = (
                self.scaler.transform(self.x.reshape(-1, C))
                .reshape(self.x.shape)
                .astype(np.float32)
            )
            self.y = (
                self.scaler.transform(self.y.reshape(-1, C))
                .reshape(self.y.shape)
                .astype(np.float32)
            )
        else:
            self.scaler = None

        # ── Zero timestamps ──────────────────────────────────────────────
        self.x_mark = np.zeros((len(self.x), self.seq_len, 6), dtype=np.float32)
        self.y_mark = np.zeros(
            (len(self.y), self.label_len + self.pred_len, 6), dtype=np.float32
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple:
        return self.x[index], self.y[index], self.x_mark[index], self.y_mark[index]

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)


def register() -> None:
    DATASET_REGISTRY.register("pre_split", Dataset_PreSplit, PreSplitParameterConfig)
