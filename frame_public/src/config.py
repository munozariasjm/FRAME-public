# pnb/config.py
import os
from dataclasses import dataclass, field
from typing import Optional, Type, Callable, Union, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class PnBConfig:
    data_directory: str = "data/calcium_odd"
    file_pattern: str = "Ca*.csv"
    save_dir: str = "../pnb_results"

    input_cols: List[str] = field(
        default_factory=lambda: [
            "Ct1S0pp", "Ct1S0np", "Ct1S0nn", "Ct3S1", "C1S0", "C3P0",
            "C1P1", "C3P1", "C3S1", "CE1", "C3P2", "c1", "c2", "c3",
            "c4", "cD", "cE",
        ]
    )
    target_cols: List[str] = field(default_factory=lambda: ["Energy ket", "M1", "E2", "M1_2BC", "Rch"])
    fidelity_col: str = "emax"

    use_parity_embeddings: bool = True
    parity_embedding_dim: int = 8

    use_shell_embeddings: bool = True
    shell_vocab_size: int =8
    shell_embedding_dim: int = 8

    use_truncation_embeddings: bool = True
    truncation_embedding_dim: int = 4
    truncation_vocab_size: int = 0

    val_size: float = 0.1
    test_size: float = 0.15
    random_state: int = 42

    num_features: int = 20               # overwritten as len(input_cols)
    num_outputs: int = 0                 # overwritten from output_specs or target_cols

    z_embedding_dim: int = 2
    n_embedding_dim: int = 16
    fidelity_embedding_dim: int = 8
    n_encoding_type: str = "sinusoidal"
    max_z: int = 20
    max_n: int = 40
    max_fid: int = 10
    shared_latent_dim: int = 80
    hidden_dim: int = 80
    dropout_rate: float = 0.05

    epochs: int = 20_000
    batch_size: int = 1024 * 8
    learning_rate: float = 0.0003
    weight_decay: float = 3e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    optimizer_class: Type[optim.Optimizer] = optim.AdamW
    optimizer_kwargs: dict = field(default_factory=dict)

    scheduler_class: Optional[Type[_LRScheduler]] = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_kwargs: dict = field(default_factory=lambda: {"factor": 0.5, "patience": 200})

    loss_fn: Callable = field(default_factory=lambda: nn.MSELoss(reduction="none"))

    use_fidelity_weights: bool = True
    fidelity_weights: Dict[int, float] = field(default_factory=lambda: {4: 1.0, 6: 1.25, 8: 1.5, 10: 1.75})

    use_uncertainty_weighting: bool = True
    init_log_sigma_E: float = 0.0
    init_log_sigma_Obs: float = 0.0

    early_stopping_patience: int = 2_000
    early_stopping_min_delta: float = 1e-6
    max_grad_norm: Optional[float] = 5.0

    eig_jitter: float = 1e-6

    smooth_overlap_lambda: float = 0.2
    smooth_overlap_k: int = 2
    smooth_overlap_sigma: float = 1.2
    param_l2_lambda: float = 1e-3

    basis_l2_lambda: float = 1e-4
    symmetry_lambda: float = 0.0
    zero_trace_lambda: float = 1e-3
    basis_gram_lambda: float = 1e-3
    enforce_zero_trace_in_forward: bool = False

    use_amp: bool = False
    amp_dtype: Union[str, torch.dtype] = "bf16"

    # Outputs
    output_specs: List[Dict] = field(default_factory=list)

    observable_use_soft_projector: bool = True
    observable_soft_proj_tau: float = 0.8
    observable_soft_proj_topk: int = 3

    # matrix sizes per fidelity (optional)
    fidelity_size_map: Dict[int, int] = field(default_factory=dict)
    matrix_size_base: int = 6
    matrix_size_delta: int = 2

    # Feature lift (keep disabled to remain affine in LECs)
    use_poly_lift: bool = False
    poly_degree: int = 1
    poly_lec_index: int = 0

    shuffle_within_fidelity: bool = True
    drop_last: bool = False

    scaler_X_path: str = field(init=False)
    scaler_y_path: str = field(init=False)
    fidelity_map_path: str = field(init=False)
    truncation_map_path: str = field(init=False)

    def __post_init__(self):
        self.num_features = len(self.input_cols)
        if not self.output_specs:
            self.output_specs = []
            for name in self.target_cols:
                lname = name.lower()
                if ("energy" in lname) or lname.startswith("en"):
                    self.output_specs.append({"name": name, "type": "energy", "level": 0})
                else:
                    self.output_specs.append({"name": name, "type": "observable", "level": 0, "psd": True})
        self.num_outputs = len(self.output_specs)

        os.makedirs(self.save_dir, exist_ok=True)
        self.scaler_X_path = os.path.join(self.save_dir, "scaler_X.pkl")
        self.scaler_y_path = os.path.join(self.save_dir, "scaler_y.pkl")
        self.fidelity_map_path = os.path.join(self.save_dir, "fidelity_map.pkl")
        self.truncation_map_path = os.path.join(self.save_dir, "truncation_map.pkl")
