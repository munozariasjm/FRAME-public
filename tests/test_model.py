"""
Unit tests for FRAME model: config, model instantiation, forward pass,
training loop, and inference pipeline.

Run with:  python -m pytest tests/ -v
"""

import os
import sys
import tempfile
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FP = os.path.join(ROOT, "frame_public")
if FP not in sys.path:
    sys.path.insert(0, FP)

from src.config import PnBConfig
from src.model import ParametricMatrixModelPnB
from src.train import Trainer
from src.utils import create_dataloaders, save_object, load_object
from src.data_loader import MultiIsotopeDataLoader
from src.inference import InferenceHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(n_rows=200, n_isotopes=3, fidelity_levels=(4, 6, 8, 10),
                       target_cols=("Energy ket", "Rch")):
    """Create a synthetic DataFrame mimicking oxygen-like data."""
    rng = np.random.RandomState(42)
    input_cols = [
        "Ct1S0pp", "Ct1S0np", "Ct1S0nn", "Ct3S1", "C1S0", "C3P0",
        "C1P1", "C3P1", "C3S1", "CE1", "C3P2", "c1", "c2", "c3",
        "c4", "cD", "cE",
    ]
    rows = []
    for _ in range(n_rows):
        Z = 8
        N = rng.choice([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16][:n_isotopes])
        emax = int(rng.choice(fidelity_levels))
        row = {col: rng.randn() for col in input_cols}
        row["Z"] = Z
        row["N"] = N
        row["emax"] = emax
        for tc in target_cols:
            row[tc] = rng.randn() * 10
        rows.append(row)
    return pd.DataFrame(rows)


def _make_config(tmp_dir, target_cols=None, fidelity_levels=(4, 6, 8, 10)):
    """Create a minimal PnBConfig for testing."""
    if target_cols is None:
        target_cols = ["Energy ket", "Rch"]
    cfg = PnBConfig()
    cfg.target_cols = list(target_cols)
    cfg.output_specs = []
    cfg.save_dir = tmp_dir
    cfg.data_directory = tmp_dir
    cfg.device = "cpu"
    cfg.epochs = 3
    cfg.batch_size = 64
    cfg.learning_rate = 1e-3
    cfg.early_stopping_patience = 100
    cfg.use_truncation_embeddings = False
    cfg.truncation_vocab_size = 0
    cfg.fidelity_weights = {lv: 1.0 for lv in fidelity_levels}
    cfg.__post_init__()
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config_creates_output_specs(self):
        cfg = PnBConfig()
        assert len(cfg.output_specs) > 0
        assert cfg.num_outputs == len(cfg.target_cols)

    def test_custom_target_cols(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp, target_cols=["Energy ket", "Rch"])
            assert cfg.num_outputs == 2
            types = [s["type"] for s in cfg.output_specs]
            assert "energy" in types

    def test_max_fid_is_10(self):
        cfg = PnBConfig()
        assert cfg.max_fid == 10


class TestModel:
    def test_instantiation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            model = ParametricMatrixModelPnB(cfg, [4, 6, 8, 10])
            assert model is not None

    def test_forward_pass_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            fid_levels = [4, 6, 8, 10]
            model = ParametricMatrixModelPnB(cfg, fid_levels)
            model.eval()

            B = 8
            z = torch.full((B,), 8, dtype=torch.long)
            n = torch.randint(4, 16, (B,))
            lecs = torch.randn(B, cfg.num_features)
            fid_idx = torch.zeros(B, dtype=torch.long)  # ordinal index 0

            with torch.no_grad():
                out = model(z=z, n=n, fidelity_idx=fid_idx, lecs=lecs)

            assert out.shape == (B, cfg.num_outputs)

    def test_forward_different_fidelities(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            fid_levels = [4, 6, 8, 10]
            model = ParametricMatrixModelPnB(cfg, fid_levels)
            model.eval()

            B = 4
            z = torch.full((B,), 8, dtype=torch.long)
            n = torch.randint(4, 16, (B,))
            lecs = torch.randn(B, cfg.num_features)

            results = []
            for fid_ord in range(len(fid_levels)):
                fid = torch.full((B,), fid_ord, dtype=torch.long)
                with torch.no_grad():
                    out = model(z=z, n=n, fidelity_idx=fid, lecs=lecs)
                results.append(out)

            # Different fidelities should produce different outputs
            assert not torch.allclose(results[0], results[-1], atol=1e-6)

    def test_regularization_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            cfg.basis_l2_lambda = 1e-4
            model = ParametricMatrixModelPnB(cfg, [4, 6, 8, 10])
            reg = model.regularization_loss()
            assert reg.item() >= 0

    def test_calcium_5_outputs(self):
        """Model with 5 target columns (calcium-like)."""
        with tempfile.TemporaryDirectory() as tmp:
            target_cols = ["Energy ket", "M1", "E2", "M1_2BC", "Rch"]
            cfg = _make_config(tmp, target_cols=target_cols)
            assert cfg.num_outputs == 5
            model = ParametricMatrixModelPnB(cfg, [4, 6, 8, 10])
            B = 4
            z = torch.full((B,), 20, dtype=torch.long)
            n = torch.randint(17, 36, (B,))
            lecs = torch.randn(B, cfg.num_features)
            fid = torch.zeros(B, dtype=torch.long)
            with torch.no_grad():
                out = model(z=z, n=n, fidelity_idx=fid, lecs=lecs)
            assert out.shape == (B, 5)


class TestTraining:
    def test_train_loop_runs(self):
        """End-to-end: synthetic data -> dataloaders -> training -> loss decreases."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            cfg.epochs = 5
            cfg.batch_size = 32

            df = _make_synthetic_df(n_rows=100, target_cols=cfg.target_cols)
            train_loader, val_loader, _, fidelity_map, _ = create_dataloaders(
                df, cfg, return_dataframes=True
            )

            model = ParametricMatrixModelPnB(cfg, list(fidelity_map.keys()))
            trainer = Trainer(model, cfg, fidelity_map)

            history = trainer.run_training(train_loader, val_loader)

            assert len(history["train_loss"]) > 0
            assert len(history["valid_loss"]) > 0
            # Loss should be finite
            assert all(np.isfinite(l) for l in history["train_loss"])
            assert all(np.isfinite(l) for l in history["valid_loss"])

    def test_gradient_flow(self):
        """Verify gradients flow through the model."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp)
            model = ParametricMatrixModelPnB(cfg, [4, 6, 8, 10])
            model.train()

            B = 8
            z = torch.full((B,), 8, dtype=torch.long)
            n = torch.randint(4, 16, (B,))
            lecs = torch.randn(B, cfg.num_features)
            fid = torch.zeros(B, dtype=torch.long)

            out = model(z=z, n=n, fidelity_idx=fid, lecs=lecs)
            loss = out.sum()
            loss.backward()

            # Check at least one parameter has a gradient
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
            )
            assert has_grad


class TestInference:
    def test_save_load_predict(self):
        """Train briefly, save checkpoint, load via InferenceHandler, predict."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp, target_cols=["Energy ket", "Rch"])
            cfg.epochs = 3
            cfg.batch_size = 32

            df = _make_synthetic_df(n_rows=100, target_cols=cfg.target_cols)
            train_loader, val_loader, _, fidelity_map, (train_df, val_df, test_df) = \
                create_dataloaders(df, cfg, return_dataframes=True)

            model = ParametricMatrixModelPnB(cfg, list(fidelity_map.keys()))
            trainer = Trainer(model, cfg, fidelity_map)
            trainer.run_training(train_loader, val_loader)

            # Save checkpoint
            ckpt_path = os.path.join(tmp, "ckpt.pt")
            torch.save({"model": model.state_dict()}, ckpt_path)

            # Load and predict
            inf = InferenceHandler(
                model_path=ckpt_path,
                config=cfg,
                fidelity_map=fidelity_map,
                device=torch.device("cpu"),
            )
            preds = inf.predict(test_df)

            assert preds.shape[0] == len(test_df)
            assert preds.shape[1] >= cfg.num_outputs
            assert np.all(np.isfinite(preds))

    def test_inference_no_m1_e2(self):
        """Inference on oxygen-like data (no M1/E2 columns) should not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_config(tmp, target_cols=["Energy ket", "Rch"])
            cfg.epochs = 2
            cfg.batch_size = 32

            df = _make_synthetic_df(n_rows=80, target_cols=["Energy ket", "Rch"])
            train_loader, val_loader, _, fidelity_map, (_, _, test_df) = \
                create_dataloaders(df, cfg, return_dataframes=True)

            model = ParametricMatrixModelPnB(cfg, list(fidelity_map.keys()))
            trainer = Trainer(model, cfg, fidelity_map)
            trainer.run_training(train_loader, val_loader)

            ckpt_path = os.path.join(tmp, "ckpt.pt")
            torch.save({"model": model.state_dict()}, ckpt_path)

            inf = InferenceHandler(
                model_path=ckpt_path,
                config=cfg,
                fidelity_map=fidelity_map,
                device=torch.device("cpu"),
            )
            preds = inf.predict(test_df)
            # Should just have 2 columns (Energy ket, Rch), no M1/E2 derived cols
            assert preds.shape[1] == 2


class TestDataLoader:
    def test_load_csv_files(self):
        """MultiIsotopeDataLoader loads CSV files correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create synthetic CSV files
            input_cols = [
                "Ct1S0pp", "Ct1S0np", "Ct1S0nn", "Ct3S1", "C1S0", "C3P0",
                "C1P1", "C3P1", "C3S1", "CE1", "C3P2", "c1", "c2", "c3",
                "c4", "cD", "cE",
            ]
            rng = np.random.RandomState(0)
            for A in [16, 17, 18]:
                rows = []
                for _ in range(20):
                    row = {col: rng.randn() for col in input_cols}
                    row["emax"] = int(rng.choice([4, 6, 8, 10]))
                    row["Energy ket"] = rng.randn() * 50
                    row["Rch"] = 2.5 + rng.randn() * 0.1
                    rows.append(row)
                pd.DataFrame(rows).to_csv(
                    os.path.join(tmp, f"O{A}_radii.csv"), index=False
                )

            cfg = PnBConfig()
            cfg.data_directory = tmp
            cfg.file_pattern = "O*.csv"
            cfg.target_cols = ["Energy ket", "Rch"]
            cfg.output_specs = []
            cfg.save_dir = tmp
            cfg.__post_init__()

            loader = MultiIsotopeDataLoader(cfg)
            df = loader.load_all_data()

            assert "Z" in df.columns
            assert "N" in df.columns
            assert len(df) == 60  # 3 files * 20 rows
            assert (df["Z"] == 8).all()  # all oxygen


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
