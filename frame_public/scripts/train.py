import os, json, time, socket, random, signal, argparse, shutil, pathlib, sys
from dataclasses import asdict
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import re
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

path2here = os.path.dirname(os.path.abspath(__file__))
path2src = os.path.abspath(os.path.join(path2here, ".."))
if path2src not in sys.path:
    sys.path.insert(0, path2src)

from src.config import PnBConfig
from src.model import ParametricMatrixModelPnB
from src.train import Trainer
from src.utils import create_dataloaders, save_object, load_object
from src.data_loader import MultiIsotopeDataLoader
from src.inference import InferenceHandler

print("CUDA available:", torch.cuda.is_available())
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.stdout.flush()
print("TF32 enabled:", torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32)
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("CUDA available:", torch.cuda.is_available())

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def slurm_meta():
    env = os.environ
    return {
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "slurm_array_task_id": env.get("SLURM_ARRAY_TASK_ID"),
        "slurm_partition": env.get("SLURM_JOB_PARTITION"),
        "node": socket.gethostname(),
    }

def _json_default(o):
    try:
        if isinstance(o, torch.dtype):    return str(o)          # e.g. "torch.float32"
        if isinstance(o, torch.device):   return str(o)          # e.g. "cuda" / "cpu"
        if isinstance(o, np.generic):     return o.item()
        if isinstance(o, set):            return list(o)
        if isinstance(o, pathlib.Path):   return str(o)
    except Exception:
        pass
    if hasattr(o, "__module__") and hasattr(o, "__name__"):
        return f"{o.__module__}.{o.__name__}"
    if callable(o):
        return getattr(o, "__name__", str(o))
    return str(o)

def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)


_PREEMPT_FLAG = {"save": False}
def _sigterm_handler(signum, frame):
    print("[Signal] Received SIGTERM — will save checkpoint at epoch end.")
    _PREEMPT_FLAG["save"] = True


_PERIODIC = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,
    "Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,
    "Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,
    "Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,
}

def parse_isotope_token(tok: str) -> tuple[int, int]:
    s = tok.strip()
    if not s:
        raise ValueError("Empty isotope token")

    m = re.fullmatch(r'([A-Z][a-z]?)\s*-\s*(\d+)', s)
    if m:
        sym, A = m.group(1), int(m.group(2))
        if sym not in _PERIODIC:
            raise ValueError(f"Unknown element symbol '{sym}' in '{tok}'")
        Z = int(_PERIODIC[sym])
        N = A - Z
        if N < 0:
            raise ValueError(f"Computed N<0 for token '{tok}'")
        return Z, N

    m = re.fullmatch(r'Z\s*=\s*(\d+)\s*[, ]+\s*N\s*=\s*(\d+)', s)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.fullmatch(r'(\d+)\s*:\s*(\d+)', s)
    if m:
        return int(m.group(1)), int(m.group(2))

    raise ValueError(f"Unrecognized isotope token format: '{tok}'")


def parse_leaveout_arg(arg: str) -> list[tuple[int, int]]:
    if not arg or not arg.strip():
        return []
    pairs: list[tuple[int, int]] = []
    clauses = [c for c in re.split(r';', arg) if c.strip()]

    for clause in clauses:
        s = clause
        for m in re.finditer(r'([A-Z][a-z]?)\s*-\s*(\d+)', s):
            sym, A = m.group(1), int(m.group(2))
            if sym in _PERIODIC:
                Z = int(_PERIODIC[sym]); N = A - Z
                if N >= 0: pairs.append((Z, N))
        s = re.sub(r'([A-Z][a-z]?)\s*-\s*\d+', ' ', s)

        for m in re.finditer(r'Z\s*=\s*(\d+)\s*[, ]+\s*N\s*=\s*(\d+)', s):
            pairs.append((int(m.group(1)), int(m.group(2))))
        s = re.sub(r'Z\s*=\s*\d+\s*[, ]+\s*N\s*=\s*\d+', ' ', s)

        for m in re.finditer(r'(\d+)\s*:\s*(\d+)', s):
            pairs.append((int(m.group(1)), int(m.group(2))))
        s = re.sub(r'\d+\s*:\s*\d+', ' ', s)

        pending_Z = None
        for m in re.finditer(r'(Z\s*=\s*(\d+))|(N\s*=\s*(\d+))', s):
            if m.group(2) is not None:
                pending_Z = int(m.group(2))
            elif m.group(4) is not None:
                if pending_Z is None:
                    pending_Z = ('_PENDING_N_', int(m.group(4)))
                else:
                    if isinstance(pending_Z, tuple) and pending_Z[0] == '_PENDING_N_':
                        _ = int(m.group(4))
                    else:
                        pairs.append((int(pending_Z), int(m.group(4))))
                        pending_Z = None

    seen = set()
    out: list[tuple[int, int]] = []
    for z, n in pairs:
        key = (int(z), int(n))
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out

def apply_leaveout_policy(
    df: pd.DataFrame,
    leaveout_pairs: List[Tuple[int,int]],
    only_low_fidelity: bool,
    low_fid_value: Optional[int],
    fid_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not leaveout_pairs:
        return df.copy(), pd.DataFrame(columns=df.columns)

    df = df.copy()
    Z_vals = df["Z"].to_numpy()
    N_vals = df["N"].to_numpy()

    iso_mask = np.zeros(len(df), dtype=bool)
    for (Zt, Nt) in leaveout_pairs:
        iso_mask |= ((Z_vals == Zt) & (N_vals == Nt))

    if not iso_mask.any():
        return df, pd.DataFrame(columns=df.columns)

    if not only_low_fidelity:
        heldout_df = df.loc[iso_mask].copy()
        keep_df = df.loc[~iso_mask].copy()
        return keep_df, heldout_df

    if low_fid_value is None:
        low_fid_value = int(df.loc[iso_mask, fid_col].min())

    is_low = (df[fid_col].astype(int).to_numpy() == int(low_fid_value))
    heldout_mask = iso_mask & (~is_low)
    heldout_df = df.loc[heldout_mask].copy()

    keep_mask = (~iso_mask) | (iso_mask & is_low)
    keep_df = df.loc[keep_mask].copy()

    return keep_df, heldout_df

def _parse_op_and_rhs(expr: str):
    expr = expr.strip()
    for op in [">=", "<=", "==", "!=", ">", "<"]:
        if op in expr:
            _, rhs = expr.split(op, 1)
            rhs = rhs.strip()
            try:
                return op, int(rhs)
            except ValueError:
                raise ValueError(f"Right-hand side must be int in '{expr}'")
    if expr.startswith("[") and expr.endswith("]"):
        items = [s.strip() for s in expr[1:-1].split(",") if s.strip()]
        try:
            vals = [int(x) for x in items]
        except ValueError:
            raise ValueError(f"List must contain ints in '{expr}'")
        return "in", vals
    raise ValueError(f"Unsupported truncation rule syntax: '{expr}'")

def parse_leaveout_trunc_arg(arg: str):
    spec = {"global": None, "per_iso": {}}
    if not arg.strip():
        return spec

    parts = [p.strip() for chunk in arg.split(";") for p in chunk.split(",") if p.strip()]
    for part in parts:
        split_at = None
        for token in [">=", "<=", "==", "!=", ">", "<", "["]:
            loc = part.find(token)
            if loc != -1:
                split_at = loc
                break
        if split_at is None:
            raise ValueError(f"Cannot parse truncation rule: '{part}'")

        lhs = part[:split_at].strip()
        rhs = part[split_at:].strip()

        op_rhs = _parse_op_and_rhs(rhs)
        if lhs.upper() == "GLOBAL":
            spec["global"] = op_rhs
        else:
            Z, N = parse_isotope_token(lhs)
            spec["per_iso"][(Z, N)] = op_rhs
    return spec

def _match_trunc_condition(values: np.ndarray, cond):
    op, rhs = cond
    if op == "in":
        mask = np.isin(values, np.asarray(rhs, dtype=int))
    elif op == "==":
        mask = (values == int(rhs))
    elif op == "!=":
        mask = (values != int(rhs))
    elif op == ">":
        mask = (values > int(rhs))
    elif op == ">=":
        mask = (values >= int(rhs))
    elif op == "<":
        mask = (values < int(rhs))
    elif op == "<=":
        mask = (values <= int(rhs))
    else:
        raise ValueError(f"Unknown op '{op}'")
    return mask

def apply_truncation_leaveout_policy(
    df: pd.DataFrame,
    trunc_col: str,
    trunc_spec: Dict,
    only_low_for_isotopes: bool = False,
    low_trunc_value: Optional[int] = None,
    isotopes_scope: Optional[List[Tuple[int,int]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trunc_col not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns)

    keep_mask = np.ones(len(df), dtype=bool)
    held_mask = np.zeros(len(df), dtype=bool)
    Z = df["Z"].to_numpy()
    N = df["N"].to_numpy()
    T = df[trunc_col].astype(int).to_numpy()

    if trunc_spec.get("global") is not None:
        m = _match_trunc_condition(T, trunc_spec["global"])
        held_mask |= m
        keep_mask &= ~m

    for (Zi, Ni), cond in trunc_spec.get("per_iso", {}).items():
        iso = (Z == Zi) & (N == Ni)
        if not np.any(iso):
            continue
        m = _match_trunc_condition(T, cond)
        held_mask |= (iso & m)
        keep_mask &= ~(iso & m)

    if only_low_for_isotopes and (low_trunc_value is not None):
        if isotopes_scope is None or len(isotopes_scope) == 0:
            iso_mask = np.ones_like(keep_mask, dtype=bool)
        else:
            iso_mask = np.zeros_like(keep_mask, dtype=bool)
            for Zi, Ni in isotopes_scope:
                iso_mask |= ((Z == Zi) & (N == Ni))
        drop_mask = iso_mask & (T != int(low_trunc_value))
        held_mask |= drop_mask
        keep_mask &= ~drop_mask

    keep_df = df.loc[keep_mask].copy()
    held_df = df.loc[held_mask].copy()
    return keep_df, held_df


# ------------------------------
# CLI
# ------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Train ParametricMatrixModelPnB with leave-out controls")

    # Data
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--file_pattern", type=str, default=None)
    p.add_argument("--save_root", type=str, default="experiments")
    p.add_argument("--exp_name", type=str, default="")

    # Splits & batching
    p.add_argument("--val_size", type=float, default=None)
    p.add_argument("--test_size", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--drop_last", action="store_true")
    p.add_argument("--no_shuffle_within_fidelity", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw","adam","sgd"])
    p.add_argument("--scheduler", type=str, default="plateau", choices=["none","plateau"])
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--min_delta", type=float, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    # p.add_argument("--amp", action="store_true")
    # p.add_argument("--amp", type=bool, default=False)
    # p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16","fp16"])

    # BANNANE/PMM model dims
    p.add_argument("--z_dim", type=int, default=None)
    p.add_argument("--n_dim", type=int, default=None)
    p.add_argument("--fid_dim", type=int, default=None)
    p.add_argument("--latent_dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--matrix_size_base", type=int, default=None)
    p.add_argument("--matrix_size_delta", type=int, default=None)

    # Outputs
    p.add_argument("--obs_level", type=int, default=0)
    p.add_argument("--energy_level", type=int, default=0)
    p.add_argument("--obs_psd", action="store_true")

    # Fidelity weights
    p.add_argument("--use_fidelity_weights", action="store_true")
    p.add_argument("--fid_w_4", type=float, default=0.5)
    p.add_argument("--fid_w_6", type=float, default=1.0)
    p.add_argument("--fid_w_8", type=float, default=1.5)
    p.add_argument("--fid_w_10", type=float, default=2.5)

    # Regularizers
    p.add_argument("--smooth_overlap_lambda", type=float, default=None)
    p.add_argument("--basis_l2_lambda", type=float, default=None)
    p.add_argument("--zero_trace_lambda", type=float, default=None)
    p.add_argument("--basis_gram_lambda", type=float, default=None)
    p.add_argument("--symmetry_lambda", type=float, default=None)

    # Soft projector tuning: False by default
    p.add_argument("--soft_proj", action="store_true")
    p.add_argument("--soft_tau", type=float, default=1.0)
    p.add_argument("--soft_topk", type=int, default=0)

    # Poly lift
    p.add_argument("--poly", action="store_true")
    p.add_argument("--poly_degree", type=int, default=2)
    p.add_argument("--poly_lec_index", type=int, default=0)

    # Leave-out / extrapolation controls (isotope & fidelity)
    p.add_argument("--leaveout_isotope", type=str, default="", help="e.g. 'O-15' or 'Z=8,N=7' or multiple: 'O-15, O-22'")
    p.add_argument("--only_low_fidelity", action="store_true", help="Keep only low-fidelity for left-out isotopes in training")
    p.add_argument("--low_fid_value", type=int, default=4, help="Which fidelity value to keep when only_low_fidelity is set")
    p.add_argument("--fidelity_col", type=str, default=None)

    # Truncation leave-out / extrapolation
    p.add_argument("--leaveout_trunc", type=str, default="", help="Rules: 'GLOBAL>1; O-22>0; Z=13,N=14:[2,3]'")
    p.add_argument("--only_low_truncation", action="store_true",
                   help="For isotopes in --leaveout_isotope (or all if none given), keep only low_trunc_value and hold out the rest.")
    p.add_argument("--low_trunc_value", type=int, default=0, help="Low truncation id to keep when only_low_truncation is set")

    p.add_argument("--resume", type=str, default="")
    p.add_argument("--eval_heldout", action="store_true", help="After training, evaluate on held-out set and save metrics")

    p.add_argument("--target_cols", type=str, default="",
                   help="Comma-separated target columns (e.g. 'Energy ket,Rch'). "
                        "Defaults to config.py setting.")

    p.add_argument("--input_cols", type=str, default="",
                   help="Comma-separated input feature columns (e.g. 'Ct1S0pp,c1'). "
                        "Defaults to the 17 chiral LECs in config.py.")

    p.add_argument("--seed", type=int, default=42)
    return p


# ------------------------------
# Main
# ------------------------------

def main():
    signal.signal(signal.SIGTERM, _sigterm_handler)
    args = build_parser().parse_args()
    set_all_seeds(args.seed)

    cfg = PnBConfig()
    if args.input_cols:
        cfg.input_cols = [c.strip() for c in args.input_cols.split(",")]
        cfg.__post_init__()
    if args.target_cols:
        cfg.target_cols = [c.strip() for c in args.target_cols.split(",")]
        cfg.output_specs = []
        cfg.__post_init__()
    if args.data_dir: cfg.data_directory = args.data_dir
    if args.file_pattern: cfg.file_pattern = args.file_pattern
    if args.fidelity_col: cfg.fidelity_col = args.fidelity_col

    if args.val_size is not None: cfg.val_size = args.val_size
    if args.test_size is not None: cfg.test_size = args.test_size
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    cfg.drop_last = bool(args.drop_last)
    cfg.shuffle_within_fidelity = not args.no_shuffle_within_fidelity

    if args.epochs is not None: cfg.epochs = args.epochs
    if args.lr is not None: cfg.learning_rate = args.lr
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.patience is not None: cfg.early_stopping_patience = args.patience
    if args.min_delta is not None: cfg.early_stopping_min_delta = args.min_delta
    if args.max_grad_norm is not None: cfg.max_grad_norm = args.max_grad_norm
    if args.device is not None: cfg.device = args.device
    cfg.use_amp = False
    # cfg.amp_dtype = args.amp_dtype

    # Optimizer / scheduler
    if args.optimizer == "adam": cfg.optimizer_class = torch.optim.Adam
    elif args.optimizer == "sgd": cfg.optimizer_class = torch.optim.SGD
    else: cfg.optimizer_class = torch.optim.AdamW
    if args.scheduler == "none": cfg.scheduler_class = None
    else: cfg.scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau

    # Embeddings / latent
    if args.z_dim is not None: cfg.z_embedding_dim = args.z_dim
    if args.n_dim is not None: cfg.n_embedding_dim = args.n_dim
    if args.fid_dim is not None: cfg.fidelity_embedding_dim = args.fid_dim
    if args.latent_dim is not None: cfg.shared_latent_dim = args.latent_dim
    if args.dropout is not None: cfg.dropout_rate = args.dropout

    # PMM sizes
    if args.matrix_size_base is not None: cfg.matrix_size_base = args.matrix_size_base
    if args.matrix_size_delta is not None: cfg.matrix_size_delta = args.matrix_size_delta

    print("Outputs:", cfg.output_specs)
    print("cfg.target_cols:", cfg.target_cols)
    print("Num outputs:", cfg.num_outputs)
    print("Fidelity weights:", cfg.fidelity_weights)

    # Fidelity weights
    if args.use_fidelity_weights:
        cfg.use_fidelity_weights = True
        fw = dict(cfg.fidelity_weights)
        if args.fid_w_4 is not None: fw[4]=args.fid_w_4
        if args.fid_w_6 is not None: fw[6]=args.fid_w_6
        if args.fid_w_8 is not None: fw[8]=args.fid_w_8
        if args.fid_w_10 is not None: fw[10]=args.fid_w_10
        cfg.fidelity_weights = fw
    else:
        cfg.use_fidelity_weights = False

    # Regularizers
    for name, val in [
        ("smooth_overlap_lambda", args.smooth_overlap_lambda),
        ("basis_l2_lambda", args.basis_l2_lambda),
        ("zero_trace_lambda", args.zero_trace_lambda),
        ("basis_gram_lambda", args.basis_gram_lambda),
        ("symmetry_lambda", args.symmetry_lambda),
    ]:
        if val is not None:
            setattr(cfg, name, val)

    # Soft projector
    if args.soft_proj: cfg.observable_use_soft_projector = True
    if args.soft_tau is not None: cfg.observable_soft_proj_tau = args.soft_tau
    if args.soft_topk is not None: cfg.observable_soft_proj_topk = args.soft_topk

    # Poly lift
    if args.poly:
        cfg.use_poly_lift = True
        cfg.poly_degree = int(args.poly_degree)
        cfg.poly_lec_index = int(args.poly_lec_index)

    # Folder
    tag = args.exp_name or "default"
    run_id = f"{timestamp()}__{tag}"
    save_root = pathlib.Path(args.save_root).absolute()
    run_dir = save_root / run_id
    os.makedirs(run_dir, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    cfg.save_dir = str(run_dir)
    cfg.scaler_X_path = str(run_dir / "scaler_X.pkl")
    cfg.scaler_y_path = str(run_dir / "scaler_y.pkl")
    cfg.fidelity_map_path = str(run_dir / "fidelity_map.pkl")
    cfg.truncation_map_path = str(run_dir / "truncation_map.pkl")

    dump_json(asdict(cfg), run_dir / "config.json")
    dump_json(slurm_meta(), run_dir / "slurm_meta.json")

    # -----------------------------
    # Data loading
    # -----------------------------
    loader = MultiIsotopeDataLoader(cfg)
    combined_df: pd.DataFrame = loader.load_all_data()

    # Auto-set truncation vocab if enabled and not set
    if cfg.use_truncation_embeddings and cfg.truncation_vocab_size <= 0 and "truncation" in combined_df.columns:
        combined_df["truncation"] = pd.to_numeric(combined_df["truncation"], errors="coerce").astype(int)
        cfg.truncation_vocab_size = int(combined_df["truncation"].max()) + 1
        # Update config on disk to reflect the inferred vocab size
        dump_json(asdict(cfg), run_dir / "config.json")

    # Leave-out controls
    leave_pairs = parse_leaveout_arg(args.leaveout_isotope)
    trunc_spec = parse_leaveout_trunc_arg(args.leaveout_trunc)

    keep_df = combined_df
    heldout_df_list = []

    # Isotope leave-out
    if leave_pairs:
        keep_df, held_iso = apply_leaveout_policy(
            keep_df,
            leaveout_pairs=leave_pairs,
            only_low_fidelity=bool(args.only_low_fidelity),
            low_fid_value=int(args.low_fid_value) if args.only_low_fidelity else None,
            fid_col=cfg.fidelity_col,
        )
        heldout_df_list.append(held_iso)
        with open(run_dir / "leaveout_isotopes.txt","w") as f:
            f.write(f"leaveout_pairs={leave_pairs}\n")
            f.write(f"only_low_fidelity={bool(args.only_low_fidelity)}\n")
            if args.only_low_fidelity:
                f.write(f"low_fid_value={int(args.low_fid_value)}\n")

    # Truncation leave-out
    if args.leaveout_trunc.strip() or bool(args.only_low_truncation):
        scope_isos = leave_pairs if len(leave_pairs) > 0 else None
        keep_df, held_trunc = apply_truncation_leaveout_policy(
            keep_df,
            trunc_col="truncation",
            trunc_spec=trunc_spec,
            only_low_for_isotopes=bool(args.only_low_truncation),
            low_trunc_value=int(args.low_trunc_value) if args.only_low_truncation else None,
            isotopes_scope=scope_isos,
        )
        heldout_df_list.append(held_trunc)
        with open(run_dir / "leaveout_trunc.txt","w") as f:
            f.write(f"leaveout_trunc={args.leaveout_trunc}\n")
            f.write(f"only_low_truncation={bool(args.only_low_truncation)}\n")
            if args.only_low_truncation:
                f.write(f"low_trunc_value={int(args.low_trunc_value)}\n")

    # Aggregate held-out slices robustly
    heldout_nonempty = []
    for i, h in enumerate(heldout_df_list):
        if isinstance(h, pd.DataFrame) and not h.empty:
            heldout_nonempty.append(h)
            print(f"[LeaveOut] slice#{i}: {len(h)} rows")
        else:
            print(f"[LeaveOut] slice#{i}: 0 rows")

    if len(heldout_nonempty) > 0:
        heldout_df = pd.concat(heldout_nonempty, ignore_index=True)
    else:
        heldout_df = pd.DataFrame(columns=combined_df.columns)

    # Persist even if empty, so downstream never crashes
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    heldout_csv = art_dir / "heldout.csv"
    heldout_df.to_csv(heldout_csv, index=False)
    print(f"[LeaveOut] Removed rows: {len(heldout_df)}; kept rows: {len(keep_df)}")
    if len(heldout_df) == 0:
        print("[LeaveOut] Note: no rows were held out by the chosen policies "
            "(check --leaveout_trunc syntax and data coverage).")

    print(f"[LeaveOut] Removed rows: {len(heldout_df)}; kept rows: {len(keep_df)}")
    if len(heldout_df) == 0:
        print("[LeaveOut] Warning: no rows were held out by the chosen policies.")

    # -----------------------------
    # Dataloaders + scalers
    # -----------------------------
    train_loader, val_loader, test_loader, fidelity_map, dfs = create_dataloaders(
        keep_df, cfg, return_dataframes=True
    )

    # Save scalers into THIS run folder
    scaler_X = None
    scaler_y = None
    if hasattr(train_loader, "dataset"):
        ds = train_loader.dataset
        scaler_X = getattr(ds, "scaler_X", None)
        scaler_y = getattr(ds, "scaler_y", None)

    if scaler_X is None:
        scaler_X = getattr(loader, "scaler_X", None)
    if scaler_y is None:
        scaler_y = getattr(loader, "scaler_y", None)

    if scaler_X is None or scaler_y is None:
        print("[WARN] Could not find fitted scalers to save — evaluation will not find them either.")
    else:
        save_object(scaler_X, cfg.scaler_X_path)
        save_object(scaler_y, cfg.scaler_y_path)
        print(f"[Scalers] Saved to:\n  - {cfg.scaler_X_path}\n  - {cfg.scaler_y_path}")

    # Persist split CSVs
    train_df, val_df, test_df = dfs
    train_df.to_csv(run_dir / "artifacts" / "train.csv", index=False)
    val_df.to_csv(run_dir / "artifacts" / "val.csv", index=False)
    test_df.to_csv(run_dir / "artifacts" / "test.csv", index=False)

    # Fidelity map (for eval)
    with open(run_dir / "fidelity_map.json","w") as f:
        json.dump({int(k): int(v) for k,v in fidelity_map.items()}, f, indent=2, sort_keys=True)

    # -----------------------------
    # Model + training
    # -----------------------------
    model = ParametricMatrixModelPnB(cfg, list(fidelity_map.keys())).to(cfg.device)
    trainer = Trainer(model, cfg, fidelity_map)

    # Resume
    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] Loading checkpoint: {args.resume}")
        state = torch.load(args.resume, map_location=cfg.device)
        model.load_state_dict(state["model"])
        if "optimizer" in state and "scheduler" in state:
            try:
                trainer.optimizer.load_state_dict(state["optimizer"])
                if trainer.scheduler and state["scheduler"]:
                    trainer.scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                print(f"[Resume] Optimizer/scheduler state not loaded: {e}")

    best_path = run_dir / "checkpoints" / "ckpt_best.pt"
    last_path = run_dir / "checkpoints" / "ckpt_last.pt"
    hist_path = run_dir / "history.json"

    def save_ckpt(path):
        torch.save({
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "fidelity_map": fidelity_map,
            "config": asdict(cfg),
            "time": time.time(),
        }, path)

    print(f"[Run] Starting training → {run_dir}")
    history = trainer.run_training(train_loader, val_loader)
    dump_json(history, hist_path)
    save_ckpt(last_path)
    save_ckpt(best_path)

    if _PREEMPT_FLAG["save"]:
        print("[Signal] Preemption flag set; checkpoint saved as ckpt_last.")

    if args.eval_heldout and (len(heldout_df) > 0):
        try:
            inf = InferenceHandler(
                model_path=str(best_path),
                config=cfg,
                fidelity_map=fidelity_map,
                device=torch.device(cfg.device),
            )
            preds = inf.predict(heldout_df)
            pred_df = heldout_df.copy()
            for i, spec in enumerate(cfg.output_specs):
                pred_df[f"pred_{spec['name']}"] = preds[:, i]
            pred_df.to_csv(run_dir / "artifacts" / "heldout_preds.csv", index=False)

            metrics = {}
            for i, spec in enumerate(cfg.output_specs):
                col = spec["name"]
                t = heldout_df[col].to_numpy()
                p = preds[:, i]
                rmse = float(np.sqrt(np.mean((t - p) ** 2)))
                metrics[f"RMSE_{col}"] = rmse
            dump_json(metrics, run_dir / "artifacts" / "heldout_metrics.json")
            print(f"[HeldOut] metrics: {metrics}")
        except Exception as e:
            print(f"[HeldOut] Evaluation failed: {e}")

    print("[Run] Done.")
    print(f"  - {hist_path}")
    print(f"  - {best_path}")
    print(f"  - {last_path}")

if __name__ == "__main__":
    main()
