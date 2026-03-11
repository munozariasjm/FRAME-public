# pnb/train.py

import copy
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .config import PnBConfig
from .model import ParametricMatrixModelPnB
from .utils import _squeeze_B1


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class MultiTaskWeights(torch.nn.Module):
    def __init__(self, init_log_sigma_E: float = 0.0, init_log_sigma_Obs: float = 0.0):
        super().__init__()
        self.log_sigma_E = torch.nn.Parameter(torch.tensor(float(init_log_sigma_E)))
        self.log_sigma_Obs = torch.nn.Parameter(torch.tensor(float(init_log_sigma_Obs)))

    def combine(self, E_loss: Optional[torch.Tensor], Obs_loss: Optional[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        if (E_loss is not None) and torch.isfinite(E_loss):
            wE = torch.exp(-self.log_sigma_E)
            total = total + (wE * E_loss + self.log_sigma_E)
        if (Obs_loss is not None) and torch.isfinite(Obs_loss):
            wO = torch.exp(-self.log_sigma_Obs)
            total = total + (wO * Obs_loss + self.log_sigma_Obs)
        if not torch.is_tensor(total):
            total = torch.tensor(total, dtype=torch.float32,
                                 device=E_loss.device if E_loss is not None else Obs_loss.device)
        return total


class Trainer:
    def __init__(self, model: ParametricMatrixModelPnB, config: PnBConfig, fidelity_map: Dict[int, int]):
        self.model = model
        self.config = config

        self.device = getattr(config, "device", torch.device("cpu"))
        self.model.to(self.device)

        self.fid_levels = sorted(list(fidelity_map.keys()))
        self.num_fidelities = len(self.fid_levels)
        self.registered_levels = torch.tensor(self.fid_levels, dtype=torch.float32)

        self.use_uncertainty_weighting = bool(getattr(config, "use_uncertainty_weighting", True))
        self.task_weights = MultiTaskWeights(
            init_log_sigma_E=float(getattr(config, "init_log_sigma_E", 0.0)),
            init_log_sigma_Obs=float(getattr(config, "init_log_sigma_Obs", 0.0)),
        ).to(self.device)

        opt_params = list(self.model.parameters())
        if self.use_uncertainty_weighting:
            opt_params += list(self.task_weights.parameters())

        self.optimizer = config.optimizer_class(
            opt_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            **getattr(config, "optimizer_kwargs", {})
        )
        self.scheduler = (
            config.scheduler_class(self.optimizer, **getattr(config, "scheduler_kwargs", {}))
            if getattr(config, "scheduler_class", None)
            else None
        )
        self.loss_fn = config.loss_fn

        self.max_grad_norm = getattr(config, "max_grad_norm", None)
        self.param_l2_lambda = float(getattr(config, "param_l2_lambda", 0.0))

        self.smooth_overlap_lambda = float(getattr(config, "smooth_overlap_lambda", 0.0))
        self.smooth_overlap_k = int(getattr(config, "smooth_overlap_k", 2))
        self.smooth_overlap_sigma = float(getattr(config, "smooth_overlap_sigma", 1.2))

        self.early_stopping = EarlyStopping(
            patience=getattr(config, "early_stopping_patience", 7),
            min_delta=getattr(config, "early_stopping_min_delta", 0.0),
        )

        self.fidelity_weights_tensor = None
        if bool(getattr(self.config, "use_fidelity_weights", False)):
            weights = torch.ones(self.num_fidelities, dtype=torch.float32)
            for i, level in enumerate(self.fid_levels):
                weights[i] = self.config.fidelity_weights.get(level, 1.0)
            self.fidelity_weights_tensor = weights.to(self.device)
            print(f"[Trainer] Fidelity weights (by ordinal index 0..{self.num_fidelities-1}) -> {self.fidelity_weights_tensor}")

        self.use_amp = bool(getattr(config, "use_amp", False))
        amp_dtype_str = str(getattr(config, "amp_dtype", "bf16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
        # self.scaler = torch.cuda.amp.GradScaler(
        #     enabled=self.use_amp and torch.cuda.is_available() and self.amp_dtype == torch.float16
        # )
        self.scaler = torch.amp.GradScaler("cuda" if (self.use_amp and torch.cuda.is_available() and self.amp_dtype == torch.float16) else "cpu")


        self.energy_idx = torch.tensor(self.model.energy_indices, dtype=torch.long, device=self.device)
        self.obs_idx = torch.tensor(self.model.observable_indices, dtype=torch.long, device=self.device)

        self._step_counter = 0
        self.reg_every = 10

    def _map_fidelity_to_index_tensor(self, fid_tensor: torch.Tensor) -> torch.Tensor:
        f = _squeeze_B1(fid_tensor).to(self.device)
        if torch.all((f >= 0) & (f < self.num_fidelities) & (f == f.floor())):
            return f.long()
        levels = self.registered_levels.to(self.device, f.dtype)  # [F]
        dist = torch.abs(f.unsqueeze(-1) - levels.unsqueeze(0))   # [B, F]
        idx = torch.argmin(dist, dim=1).long()
        return idx

    def _autocast_ctx(self):
        device_type = "cuda" if (hasattr(self.device, "type") and self.device.type == "cuda") else "cpu"
        return torch.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp)

    def _maybe_clip(self):
        if self.max_grad_norm is not None:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def _sched_step(self, valid_loss: float):
        if not self.scheduler:
            return
        name = self.scheduler.__class__.__name__.lower()
        if "plateau" in name:
            self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()

    def _split_group_losses(self, raw_loss: torch.Tensor, fid_ord: Optional[torch.Tensor]):
        L = raw_loss
        if (self.fidelity_weights_tensor is not None) and (fid_ord is not None):
            wB = self.fidelity_weights_tensor[fid_ord].view(-1, 1)  # [B,1]
            L = L * wB

        E_loss = None
        Obs_loss = None
        if self.energy_idx.numel() > 0:
            E_loss = L[:, self.energy_idx].mean()
        if self.obs_idx.numel() > 0:
            Obs_loss = L[:, self.obs_idx].mean()
        return E_loss, Obs_loss

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, ...]:
        """
        Return (z, n, lecs, fid, features, y, shell_opt, trunc_opt)
        Supports:
        - len==6: no shell, no trunc
        - len==7: either shell OR trunc
        - len==8: both shell AND trunc
        """
        if len(batch) == 6:
            z, n, lecs, fid, features, y = batch
            shell = None
            trunc = None
        elif len(batch) == 7:
            z, n, lecs, fid, features, y, extra = batch
            if hasattr(self.config, "use_shell_embeddings") and bool(getattr(self.config, "use_shell_embeddings", False)):
                shell = extra
                trunc = None
            else:
                shell = None
                trunc = extra
        elif len(batch) == 8:
            z, n, lecs, fid, features, y, shell, trunc = batch
        else:
            raise ValueError(f"Unexpected batch size {len(batch)}")
        return z, n, lecs, fid, features, y, shell, trunc

    def _compute_regularizers(self, batch) -> torch.Tensor:
        z, n, lecs, fid, _, y, shell, trunc = batch
        reg = torch.zeros((), device=self.device)

        # Smooth eigenvector overlap
        if (self.smooth_overlap_lambda > 0.0) and (self._step_counter % self.reg_every == 0):
            if z.size(0) >= 2:
                if z.size(0) > 64:
                    idx = torch.randperm(z.size(0), device=z.device)[:64]
                    z_sub, n_sub, lecs_sub, fid_sub = z[idx], n[idx], lecs[idx], fid[idx]
                    shell_sub = shell[idx] if (shell is not None) else None
                    trunc_sub = trunc[idx] if (trunc is not None) else None
                else:
                    z_sub, n_sub, lecs_sub, fid_sub = z, n, lecs, fid
                    shell_sub = shell
                    trunc_sub = trunc
                reg = reg + self.smooth_overlap_lambda * self.model.eigen_overlap_penalty(
                    z_sub.to(self.device),
                    n_sub.to(self.device),
                    fid_sub.to(self.device),
                    lecs_sub.to(self.device),
                    k=self.smooth_overlap_k,
                    sigma=self.smooth_overlap_sigma,
                    shell=None if shell_sub is None else shell_sub.to(self.device),
                    trunc=None if trunc_sub is None else trunc_sub.to(self.device),
                )

        # Parameter regularization
        if self.param_l2_lambda > 0.0:
            reg = reg + self.model.regularization_loss()

        return reg

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for raw_batch in train_loader:
            self._step_counter += 1
            batch = tuple(b.to(self.device) for b in raw_batch)
            z, n, lecs, fid, features, y, shell, trunc = self._unpack_batch(batch)

            fid_ord = self._map_fidelity_to_index_tensor(fid)
            assert torch.all(fid_ord == fid_ord[0]), "Bucketed DataLoader must yield one fidelity per batch."

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast_ctx():
                y_pred = self.model(
                    z=z, n=n, fidelity_idx=fid, lecs=lecs,
                    shell=None if shell is None else shell,
                    trunc=None if trunc is None else trunc
                )
                raw_loss = self.loss_fn(y_pred, y)    # elementwise [B,O]
                E_loss, Obs_loss = self._split_group_losses(raw_loss, fid_ord)

                if self.use_uncertainty_weighting:
                    data_loss = self.task_weights.combine(E_loss, Obs_loss)
                else:
                    data_loss = raw_loss.mean()

                reg = self._compute_regularizers((z, n, lecs, fid, features, y, shell, trunc))
                loss = data_loss + reg

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self._maybe_clip()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self._maybe_clip()
                self.optimizer.step()

            total_loss += float(loss.detach().item())

        return total_loss / max(1, len(train_loader))

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for raw_batch in val_loader:
            batch = tuple(b.to(self.device) for b in raw_batch)
            z, n, lecs, fid, features, y, shell, trunc = self._unpack_batch(batch)

            fid_ord = self._map_fidelity_to_index_tensor(fid)
            assert torch.all(fid_ord == fid_ord[0]), "Bucketed DataLoader must yield one fidelity per batch."

            with self._autocast_ctx():
                y_pred = self.model(
                    z=z, n=n, fidelity_idx=fid, lecs=lecs,
                    shell=None if shell is None else shell,
                    trunc=None if trunc is None else trunc
                )
                raw_loss = self.loss_fn(y_pred, y)
                E_loss, Obs_loss = self._split_group_losses(raw_loss, fid_ord)
                if self.use_uncertainty_weighting:
                    loss = self.task_weights.combine(E_loss, Obs_loss)
                else:
                    loss = raw_loss.mean()
            total_loss += float(loss.detach().item())
        return total_loss / max(1, len(val_loader))

    def run_training(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        history = {"train_loss": [], "valid_loss": [], "learning_rate": []}
        best_model_state = None

        iterator = tqdm(range(self.config.epochs), desc="Training")
        for epoch in iterator:
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.evaluate(val_loader)

            self._sched_step(valid_loss)

            history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)

            iterator.set_postfix(train_loss=f"{train_loss:.4f}", valid_loss=f"{valid_loss:.4f}")

            if self.early_stopping.best_loss is None or valid_loss < self.early_stopping.best_loss:
                best_model_state = copy.deepcopy(self.model.state_dict())

            if self.early_stopping(valid_loss):
                print(f"\n[Trainer] Early stopping at epoch {epoch + 1}")
                break

            if hasattr(self.device, "type") and self.device.type == "cuda":
                torch.cuda.empty_cache()

        if best_model_state:
            self.model.load_state_dict(best_model_state)
        return history
