"""Dataset backed by raw time-series + window-start index files.

Two formats are auto-detected based on ``raw.npy`` ndim:

**v2 format** (``raw.ndim == 2``):

- ``raw.npy``              — ``(T, C_eff)`` float32, single raw time-series
- ``train_idx.npy``        — ``(n_train,)`` int32, absolute window-start positions
- ``test_idx.npy``         — ``(n_test,)`` int32, absolute window-start positions
- ``sample_r020_idx.npy``  — int32, indices INTO ``train_idx`` (optional, 20% ratio)
- ``sample_r010_idx.npy``  — int32, indices INTO ``train_idx`` (main table, 10%)
- ``sample_r005_idx.npy``  — int32, indices INTO ``train_idx`` (optional, 5%)
- ``sample_r001_idx.npy``  — int32, indices INTO ``train_idx`` (optional, 1%)
- ``meta.json``

**v3 format** (``raw.ndim == 3``):

- ``raw.npy``              — ``(K, T_max, C)`` float32, K source series zero-padded
- ``raw_lengths.npy``      — ``(K,)`` int32, actual T_i per series
- ``train_idx.npy``        — ``(n_train, 2)`` int32, ``[series_id, start_position]`` pairs
- ``test_idx.npy``         — ``(n_test, 2)`` int32, same
- ``meta.json``

Val is carved from the tail of ``train_idx`` (chronologically latest train
windows).  StandardScaler is fitted on the raw series restricted to the time
span covered by the train windows.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from benchmark.registry import DATASET_REGISTRY
from data.schemas.datasets.raw_idx import RawIdxParameterConfig

_SAMPLE_FLAGS = {"sample_r020", "sample_r010", "sample_r005", "sample_r001"}


class Dataset_RawIdx(Dataset):
    """Raw + idx dataset.

    Parameters
    ----------
    root_path : str
        Directory containing ``raw.npy`` + ``*_idx.npy`` + ``meta.json``.
    data_path : str
        Unused; kept for API compatibility.
    size : tuple[int, int, int]
        ``(seq_len, label_len, pred_len)``.
    flag : str
        One of ``"train"``, ``"val"``, ``"test"``,
        ``"sample_r020"``, ``"sample_r010"``, ``"sample_r005"``, ``"sample_r001"``.
    val_ratio : float
        Fraction of ``train_idx`` reserved for validation (tail slice).
    scale : bool
        Whether to apply StandardScaler fitted on the train portion of raw.
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
        total_len = self.seq_len + self.pred_len

        raw = np.load(os.path.join(root_path, "raw.npy"), mmap_mode="r")

        if raw.ndim == 2:
            windows = self._load_v2(
                raw, root_path, flag, val_ratio, scale, total_len
            )
        elif raw.ndim == 3:
            windows = self._load_v3(
                raw, root_path, flag, val_ratio, scale, total_len
            )
        else:
            raise ValueError(
                f"raw.npy has unsupported ndim={raw.ndim}; expected 2 (v2) or 3 (v3)."
            )

        self.x = windows[:, : self.seq_len, :]
        self.y = windows[:, self.seq_len - self.label_len :, :]

        self.x_mark = np.zeros((len(self.x), self.seq_len, 6), dtype=np.float32)
        self.y_mark = np.zeros(
            (len(self.y), self.label_len + self.pred_len, 6), dtype=np.float32
        )

    def _load_v2(
        self,
        raw: np.ndarray,
        root_path: str,
        flag: str,
        val_ratio: float,
        scale: bool,
        total_len: int,
    ) -> np.ndarray:
        """Original v2 format: raw is (T, C); idx files are 1D absolute starts."""
        train_idx = np.load(os.path.join(root_path, "train_idx.npy")).astype(np.int64)
        n_all_train = len(train_idx)
        n_val = max(1, round(n_all_train * val_ratio))
        n_train_only = n_all_train - n_val

        if flag == "train":
            positions = train_idx[:n_train_only]
        elif flag == "val":
            positions = train_idx[n_train_only:]
        elif flag == "test":
            positions = np.load(os.path.join(root_path, "test_idx.npy")).astype(
                np.int64
            )
        elif flag in _SAMPLE_FLAGS:
            rel = np.load(os.path.join(root_path, f"{flag}_idx.npy")).astype(np.int64)
            positions = train_idx[rel]
        else:
            raise ValueError(
                f"Invalid flag '{flag}'. Expected train/val/test or one of {sorted(_SAMPLE_FLAGS)}."
            )

        # Fit strictly on the authentic chronological training partition.
        # train_idx also contains validation starts, so using train_idx.max()
        # would leak validation values into normalization.
        if scale and n_all_train > 0:
            meta_path = os.path.join(root_path, "meta.json")
            if os.path.isfile(meta_path):
                import json

                with open(meta_path, encoding="utf-8") as handle:
                    meta = json.load(handle)
                T_train_end = int(meta.get("train_boundary", int(0.8 * raw.shape[0])))
            else:
                T_train_end = int(0.8 * raw.shape[0])
            T_train_end = max(1, min(T_train_end, raw.shape[0]))
            C = raw.shape[1]
            self.scaler = StandardScaler()
            self.scaler.fit(np.asarray(raw[:T_train_end]).reshape(-1, C))
            raw_scaled = (
                self.scaler.transform(np.asarray(raw).reshape(-1, C))
                .reshape(raw.shape)
                .astype(np.float32)
            )
        else:
            self.scaler = None
            raw_scaled = np.asarray(raw, dtype=np.float32)

        # Materialise windows via sliding_window_view + fancy indexing.
        if len(positions) == 0:
            C = raw.shape[1]
            windows = np.empty((0, total_len, C), dtype=np.float32)
        else:
            all_wins = sliding_window_view(
                raw_scaled, window_shape=(total_len, raw_scaled.shape[1])
            )[:, 0, :, :]
            windows = np.asarray(all_wins[positions], dtype=np.float32)

        return windows

    def _load_v3(
        self,
        raw: np.ndarray,
        root_path: str,
        flag: str,
        val_ratio: float,
        scale: bool,
        total_len: int,
    ) -> np.ndarray:
        """v3 format: raw is (K, T_max, C); idx files are (N, 2) [series_id, pos]."""
        train_idx = np.load(os.path.join(root_path, "train_idx.npy")).astype(np.int64)
        n_all_train = len(train_idx)
        n_val = max(1, round(n_all_train * val_ratio))
        n_train_only = n_all_train - n_val

        if flag == "train":
            positions_2d = train_idx[:n_train_only]
        elif flag == "val":
            positions_2d = train_idx[n_train_only:]
        elif flag == "test":
            positions_2d = np.load(os.path.join(root_path, "test_idx.npy")).astype(
                np.int64
            )
        elif flag in _SAMPLE_FLAGS:
            raise ValueError(
                f"sample_r* flags are v2-only; v3 dropped these (got '{flag}')."
            )
        else:
            raise ValueError(
                f"Invalid flag '{flag}'. Expected train/val/test."
            )

        C = raw.shape[-1]

        def _materialize(idx_2d: np.ndarray) -> np.ndarray:
            out = np.empty((len(idx_2d), total_len, C), dtype=np.float32)
            for k, (s, p) in enumerate(idx_2d):
                out[k] = raw[int(s), int(p) : int(p) + total_len, :]
            return out

        windows = _materialize(positions_2d)

        if scale and n_all_train > 0:
            train_2d = train_idx[:n_train_only]
            train_arr = windows if flag == "train" else _materialize(train_2d)
            self.scaler = StandardScaler()
            self.scaler.fit(train_arr.reshape(-1, C))
            if len(windows) > 0:
                windows = (
                    self.scaler.transform(windows.reshape(-1, C))
                    .reshape(windows.shape)
                    .astype(np.float32)
                )
        else:
            self.scaler = None

        return windows

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple:
        return self.x[index], self.y[index], self.x_mark[index], self.y_mark[index]

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)


def register() -> None:
    DATASET_REGISTRY.register("raw_idx", Dataset_RawIdx, RawIdxParameterConfig)
