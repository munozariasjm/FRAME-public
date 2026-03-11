# pnb/utils.py

import os
import pickle
import torch
import math
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Iterator, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import BatchSampler


def save_object(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_object(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)

def _load_model_state_dict(model_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Loads a checkpoint or a raw state_dict from model_path, robust across PyTorch 2.6+.
    Returns a plain state_dict suitable for model.load_state_dict(...).
    """
    try:
        obj = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(model_path, map_location=device)

    # If saved as {"model": state_dict, ...}, extract it; otherwise assume it's the state_dict already.
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        # Heuristic: if looks like a state dict (tensor leaves), use as-is
        return obj
    raise RuntimeError(f"Unexpected checkpoint format at {model_path}: type={type(obj)}")



SHELL_CLOSURES = [0, 2, 8, 16, 20, 28, 32, 50, 82, 126, 184]

def compute_shell_region_from_N(n_vals: np.ndarray, max_bins: int) -> np.ndarray:
    idx = np.digitize(n_vals, SHELL_CLOSURES, right=False) - 1
    idx = np.clip(idx, 0, max_bins - 1)
    return idx

def ensure_shell_column(df: pd.DataFrame, shell_vocab_size: int, prefer_by: str = "N") -> pd.DataFrame:
    if "shell_id" in df.columns:
        df = df.copy()
        df["shell_id"] = df["shell_id"].astype(int).clip(lower=0, upper=shell_vocab_size - 1)
        return df

    df = df.copy()
    if prefer_by.upper() == "Z":
        base_vals = df["Z"].to_numpy(dtype=np.int64)
        shell = compute_shell_region_from_N(base_vals, shell_vocab_size)  # reuse magic numbers as proxy
    else:
        base_vals = df["N"].to_numpy(dtype=np.int64)
        shell = compute_shell_region_from_N(base_vals, shell_vocab_size)

    df["shell_id"] = shell.astype(np.int64)
    return df


class FidelityBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: TensorDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle_within: bool = True,
        round_robin: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle_within = bool(shuffle_within)
        self.round_robin = bool(round_robin)
        self._buckets = self._build_buckets()

    def _build_buckets(self) -> Dict[int, List[int]]:
        buckets: Dict[int, List[int]] = {}
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            fid = sample[3]
            fid_val = int(fid.item()) if torch.is_tensor(fid) else int(fid)
            buckets.setdefault(fid_val, []).append(idx)
        return buckets

    def __iter__(self) -> Iterator[List[int]]:
        bucket_keys = sorted(self._buckets.keys())
        per_bucket_lists = []
        for k in bucket_keys:
            lst = self._buckets[k][:]
            if self.shuffle_within:
                perm = torch.randperm(len(lst)).tolist()
                lst = [lst[i] for i in perm]
            per_bucket_lists.append(lst)

        per_bucket_batches = []
        for lst in per_bucket_lists:
            batches = [lst[i:i + self.batch_size] for i in range(0, len(lst), self.batch_size)]
            if self.drop_last and len(batches) > 0 and len(batches[-1]) < self.batch_size:
                batches.pop()
            per_bucket_batches.append(batches)

        if self.round_robin:
            active = True
            i = 0
            while active:
                active = False
                for b in per_bucket_batches:
                    if i < len(b):
                        yield b[i]
                        active = True
                i += 1
        else:
            for batches in per_bucket_batches:
                for b in batches:
                    yield b

    def __len__(self) -> int:
        total = 0
        for _, lst in self._buckets.items():
            nb = len(lst) // self.batch_size
            if (len(lst) % self.batch_size) and (not self.drop_last):
                nb += 1
            total += nb
        return total


def make_bucketed_loader(
    dataset: TensorDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    shuffle_within: bool = True,
    round_robin: bool = True,
) -> DataLoader:
    sampler = FidelityBucketBatchSampler(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle_within=shuffle_within,
        round_robin=round_robin,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)


def create_dataloaders(
    data: pd.DataFrame,
    config,  # PnBConfig
    return_dataframes: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    train_val_df, test_df = train_test_split(
        data,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    val_frac = config.val_size / (1 - config.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_frac,
        random_state=config.random_state,
    )
    print(f"[Preprocess] Data split => train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    if getattr(config, "use_shell_embeddings", False):
        prefer_by = "N"  # or "Z"
        train_df = ensure_shell_column(train_df, config.shell_vocab_size, prefer_by=prefer_by)
        val_df   = ensure_shell_column(val_df,   config.shell_vocab_size, prefer_by=prefer_by)
        test_df  = ensure_shell_column(test_df,  config.shell_vocab_size, prefer_by=prefer_by)
        print("[Preprocess] Shell indicators ready (column 'shell_id').")

    scaler_X = StandardScaler().fit(train_df[config.input_cols])
    scaler_y = StandardScaler().fit(train_df[config.target_cols])

    try:
        save_object(scaler_X, config.scaler_X_path)
        save_object(scaler_y, config.scaler_y_path)
    except NameError:
        pass

    all_fidelities = sorted(data[config.fidelity_col].unique())
    fidelity_map = {int(fid): i for i, fid in enumerate(all_fidelities)}
    print(f"[Preprocess] Fidelity mapping created: {fidelity_map}")

    try:
        save_object(fidelity_map, config.fidelity_map_path)
    except NameError:
        pass

    def to_tensor_dataset(df: pd.DataFrame) -> TensorDataset:
        X_scaled = StandardScaler().fit_transform(df[config.input_cols]) if False else None  # placeholder to show intent
        X_scaled = scaler_X.transform(df[config.input_cols])
        y_scaled = scaler_y.transform(df[config.target_cols])

        lecs = torch.tensor(X_scaled, dtype=torch.float32)
        features = torch.tensor(X_scaled, dtype=torch.float32)

        targets = torch.tensor(y_scaled, dtype=torch.float32)
        Z = torch.tensor(df["Z"].values, dtype=torch.long)
        N = torch.tensor(df["N"].values, dtype=torch.long)

        fid_idx = torch.tensor([fidelity_map[int(fid)] for fid in df[config.fidelity_col].values], dtype=torch.long)

        if getattr(config, "use_shell_embeddings", False):
            shell_ids = torch.tensor(df["shell_id"].values, dtype=torch.long)
            return TensorDataset(Z, N, lecs, fid_idx, features, targets, shell_ids)
        else:
            return TensorDataset(Z, N, lecs, fid_idx, features, targets)

    train_dataset = to_tensor_dataset(train_df)
    val_dataset   = to_tensor_dataset(val_df)
    test_dataset  = to_tensor_dataset(test_df)

    train_loader = make_bucketed_loader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=getattr(config, "drop_last", False),
        shuffle_within=getattr(config, "shuffle_within_fidelity", True),
        round_robin=True,
    )

    val_loader = make_bucketed_loader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle_within=False,
        round_robin=False,
    )
    test_loader = make_bucketed_loader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle_within=False,
        round_robin=False,
    )

    if return_dataframes:
        return train_loader, val_loader, test_loader, fidelity_map, (train_df, val_df, test_df)
    return train_loader, val_loader, test_loader, fidelity_map


def _squeeze_B1(x: torch.Tensor) -> torch.Tensor:
    if x.dim() > 1 and x.size(-1) == 1:
        return x.squeeze(-1)
    return x


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_idx: int):
        super().__init__()
        self.emb = nn.Embedding(int(max_idx), int(d))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx.clamp_min_(0))


class SinusoidalNEncoding(nn.Module):
    def __init__(self, d: int, max_n: int):
        super().__init__()
        self.d = int(d)
        k = torch.arange(0, d // 2, dtype=torch.float32)
        self.register_buffer("freq", torch.exp(-math.log(10_000.0) * (2 * k) / max(1, d)), persistent=False)

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        n = n.float().unsqueeze(-1)  # [B,1]
        w = self.freq.view(1, -1)
        sin = torch.sin(n * w)
        cos = torch.cos(n * w)
        x = torch.cat([sin, cos], dim=-1)
        if x.size(-1) < self.d:
            pad = torch.zeros(x.size(0), self.d - x.size(-1), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        return x
