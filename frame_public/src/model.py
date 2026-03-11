# pnb/model.py

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn

from .config import PnBConfig
from .utils import PositionalEncoding, SinusoidalNEncoding, _squeeze_B1

class FiLM(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(in_dim, hid_dim * 2),
            nn.LeakyReLU(inplace=False),
            nn.Linear(hid_dim * 2, hid_dim * 2),
        )

    def forward(self, x_cond: torch.Tensor, h_feat: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(x_cond)  # [B, 2*hid]
        g, b = gb.chunk(2, dim=-1)
        return g * h_feat + b


class BANNANE_backbone(nn.Module):
    """
    Global latent encoder for (Z, N, fidelity, parity, shell, truncation).
    """

    def __init__(self, config: PnBConfig, fidelity_levels: List[int]):
        super().__init__()
        self.config = config
        self.fidelity_levels = sorted(list(fidelity_levels))
        self.num_fidelities = len(self.fidelity_levels)

        self.register_buffer(
            "_levels_tensor",
            torch.tensor(self.fidelity_levels, dtype=torch.float32),
            persistent=False,
        )

        # --- N encoding ---
        n_encoding_type = getattr(config, "n_encoding_type", "sinusoidal").lower()
        if n_encoding_type == "sinusoidal":
            self.n_encoding = SinusoidalNEncoding(
                config.n_embedding_dim, config.max_n
            )
        else:
            self.n_encoding = PositionalEncoding(
                config.n_embedding_dim, config.max_n + 1
            )

        # --- Z encoding ---
        z_encoding_type = getattr(config, "z_encoding_type", "positional").lower()
        if z_encoding_type == "positional":
            self.z_embedding = PositionalEncoding(
                config.z_embedding_dim, config.max_z + 1
            )
        else:
            self.z_embedding = SinusoidalNEncoding(
                config.z_embedding_dim, config.max_z
            )

        self.fidelity_embedding = PositionalEncoding(
            config.fidelity_embedding_dim, self.num_fidelities
        )

        self.use_parity_embeddings = bool(
            getattr(config, "use_parity_embeddings", True)
        )
        if self.use_parity_embeddings:
            dim = int(getattr(config, "parity_embedding_dim", 4))
            self.parity_embedding = PositionalEncoding(dim, 4)
            self.parity_dim = dim
        else:
            self.parity_embedding = None
            self.parity_dim = 0

        self.use_shell_embeddings = bool(
            getattr(config, "use_shell_embeddings", False)
        )
        if self.use_shell_embeddings:
            vocab = int(getattr(config, "shell_vocab_size", 8))
            dim = int(getattr(config, "shell_embedding_dim", 4))
            self.shell_embedding = PositionalEncoding(dim, vocab)
            self.shell_dim = dim
        else:
            self.shell_embedding = None
            self.shell_dim = 0

        self.use_truncation_embeddings = bool(
            getattr(config, "use_truncation_embeddings", False)
        )
        if self.use_truncation_embeddings and int(
            getattr(config, "truncation_vocab_size", 0)
        ) > 0:
            t_vocab = int(getattr(config, "truncation_vocab_size", 0))
            t_dim = int(getattr(config, "truncation_embedding_dim", 4))
            self.trunc_embedding = PositionalEncoding(t_dim, t_vocab)
            self.trunc_dim = t_dim
        else:
            self.trunc_embedding = None
            self.trunc_dim = 0
            self.use_truncation_embeddings = False

        emb_dim = (
            config.z_embedding_dim
            + config.n_embedding_dim
            + config.fidelity_embedding_dim
            + self.parity_dim
            + self.shell_dim
            + self.trunc_dim
        )
        hid = config.shared_latent_dim

        self.pre = nn.Sequential(
            nn.Linear(emb_dim, hid),
            nn.LeakyReLU(inplace=False),
        )
        self.film = FiLM(emb_dim, hid)
        self.post = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(hid, hid),
            nn.LeakyReLU(inplace=False),
            nn.LayerNorm(hid),
        )

    @staticmethod
    def _safe_idx(x: torch.Tensor) -> torch.Tensor:
        return _squeeze_B1(x).detach().clone().to(torch.long).contiguous()

    def _map_fidelity_to_ord(self, f: torch.Tensor) -> torch.Tensor:
        f = _squeeze_B1(f).float()
        levels = self._levels_tensor.to(f.device, f.dtype)

        if torch.all((f >= 0) & (f < len(levels)) & (f == f.floor())):
            return f.long()

        dist = torch.abs(f.unsqueeze(-1) - levels.unsqueeze(0))
        idx = torch.argmin(dist, dim=1).long()
        return idx

    @staticmethod
    def _parity_class(z: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        z_par = (z & 1)
        n_par = (n & 1)
        return (z_par << 1) | n_par

    def latent(
        self,
        z: torch.Tensor,
        n: torch.Tensor,
        fidelity_idx: torch.Tensor,
        shell: Optional[torch.Tensor] = None,
        trunc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_idx = self._safe_idx(z)
        n_idx = self._safe_idx(n)
        fid_ord = self._map_fidelity_to_ord(self._safe_idx(fidelity_idx))

        parts = [
            self.z_embedding(z_idx),
            self.n_encoding(n_idx),
            self.fidelity_embedding(fid_ord),
        ]

        if self.use_parity_embeddings:
            pcls = self._parity_class(z_idx, n_idx)
            parts.append(self.parity_embedding(pcls))

        if self.use_shell_embeddings:
            if shell is None:
                B = z_idx.size(0)
                parts.append(
                    torch.zeros(
                        B,
                        self.shell_dim,
                        device=z_idx.device,
                        dtype=torch.float32,
                    )
                )
            else:
                parts.append(self.shell_embedding(self._safe_idx(shell)))

        if self.use_truncation_embeddings:
            if trunc is None:
                B = z_idx.size(0)
                parts.append(
                    torch.zeros(
                        B,
                        self.trunc_dim,
                        device=z_idx.device,
                        dtype=torch.float32,
                    )
                )
            else:
                parts.append(self.trunc_embedding(self._safe_idx(trunc)))

        x_cond = torch.cat(parts, dim=-1)
        h0 = self.pre(x_cond)
        h_mod = self.film(x_cond, h0)
        h = self.post(h_mod)
        return h, fid_ord


class ConvergenceFlow(nn.Module):
    def __init__(self, emb_dim: int, num_lecs: int, hidden_dim: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(emb_dim + num_lecs, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.gate[-2].bias.data.fill_(2.0)

    def forward(self, fid_val: float, f_min: float, h: torch.Tensor, lecs: torch.Tensor) -> torch.Tensor:
        base_coord = (1.0 / f_min) - (1.0 / max(fid_val, 1e-6))
        lecs_used = lecs[..., : min(lecs.size(-1), 100)]
        phys_in = torch.cat([h, lecs_used], dim=-1)
        scale = 0.5 + self.gate(phys_in)
        return base_coord * scale


class ParametricMatrixModelPnB(nn.Module):
    def __init__(
        self,
        config: PnBConfig,
        fidelity_levels: List[int],
        fidelity_size_map: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.config = config

        # Backbone
        self.backbone = BANNANE_backbone(config, fidelity_levels)
        self.fid_levels = sorted(list(self.backbone.fidelity_levels))
        self.f_min = float(self.fid_levels[0])
        self.num_fid = len(self.fid_levels)

        # Output specs and indices
        self.output_specs = self._build_output_specs(config)
        self.observable_indices = [i for i, s in enumerate(self.output_specs) if s["type"] == "observable"]
        self.energy_indices = [i for i, s in enumerate(self.output_specs) if s["type"] == "energy"]
        self.num_observables = len(self.observable_indices)
        self.num_lecs = int(getattr(config, "num_lecs", config.num_features))

        # Matrix sizes
        if fidelity_size_map and len(fidelity_size_map) == len(self.fid_levels):
            sizes = [int(fidelity_size_map[l]) for l in self.fid_levels]
        else:
            base = int(getattr(config, "matrix_size_base", 4))
            delta = int(getattr(config, "matrix_size_delta", 2))
            sizes = [base + i * delta for i in range(self.num_fid)]
        self.matrix_sizes = sizes
        self.ms_max = max(sizes)
        self.register_buffer("I_full", torch.eye(self.ms_max, dtype=torch.float32), persistent=False)

        # Stage masks
        proj_masks = []
        for m in sizes:
            mask = torch.zeros(self.ms_max, self.ms_max, dtype=torch.float32)
            mask[:m, :m] = 1.0
            proj_masks.append(mask)
        self.register_buffer("stage_masks", torch.stack(proj_masks, dim=0), persistent=False)

        self.nf = config.num_features
        self.emb_dim = config.shared_latent_dim
        self.poly_order = 2

        self.convergence_modulator_H = ConvergenceFlow(self.emb_dim, self.num_lecs)

        # 2. Independent Flows for each Observable
        if self.num_observables > 0:
            self.convergence_modulator_obs = nn.ModuleList([
                ConvergenceFlow(self.emb_dim, self.num_lecs)
                for _ in range(self.num_observables)
            ])
        else:
            self.convergence_modulator_obs = None

        self._base_coeff_in_dim = self.emb_dim + self.nf
        self.anchor_coeff_net = nn.Sequential(
            nn.Linear(self._base_coeff_in_dim, self.emb_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.emb_dim, self.nf + 1)
        )
        self.refine_coeff_net = nn.Sequential(
            nn.Linear(self._base_coeff_in_dim, self.emb_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.emb_dim, self.poly_order * (self.nf + 1))
        )

        def xavier(shape):
            W = nn.Parameter(torch.empty(*shape, dtype=torch.float32))
            nn.init.xavier_uniform_(W)
            return W

        self.P_base = xavier((self.nf, self.ms_max, self.ms_max))
        self.P_refine = xavier((self.poly_order, self.nf, self.ms_max, self.ms_max))

        if self.num_observables > 0:
            self.H_base = xavier((self.num_observables, self.nf + 1, self.ms_max, self.ms_max))
            self.Q_base = xavier((self.num_observables, self.nf + 1, self.ms_max, self.ms_max))
            self.H_refine = xavier((self.poly_order, self.num_observables, self.nf + 1, self.ms_max, self.ms_max))
            self.Q_refine = xavier((self.poly_order, self.num_observables, self.nf + 1, self.ms_max, self.ms_max))
            self.obs_adapters = nn.ModuleList([
                nn.Linear(self.nf + 1, self.nf + 1) for _ in range(self.num_observables)
            ])
        else:
            self.H_base = self.Q_base = self.H_refine = self.Q_refine = self.obs_adapters = None

        self.bias = nn.Parameter(torch.zeros(len(self.output_specs)))
        self.eps_jitter = float(getattr(config, "eig_jitter", 1e-6))

        # Soft Projector
        self.use_soft_projector = bool(getattr(config, "observable_use_soft_projector", True))
        self.soft_proj_tau = float(getattr(config, "observable_soft_proj_tau", 1.0))
        self.soft_proj_topk = int(getattr(config, "observable_soft_proj_topk", 0))

        # Weight for regularization
        idx = torch.arange(self.ms_max)
        I_idx, J_idx = torch.meshgrid(idx, idx, indexing="ij")
        rad = (I_idx + J_idx).float() / max(1.0, 2 * self.ms_max)
        outer_weight = 1.0 + rad
        self.register_buffer("outer_weight", outer_weight, persistent=False)


    def _build_output_specs(self, config: PnBConfig) -> List[Dict]:
        specs = getattr(config, "output_specs", None)
        if specs: return specs
        return [{"name": f"obs_{i}", "type": "observable", "level": 0, "psd": True} for i in range(config.num_outputs)]

    @staticmethod
    def _symmetrize(A: torch.Tensor) -> torch.Tensor:
        return 0.5 * (A + A.transpose(-1, -2))

    def _infer_fid_value(self, fidelity_idx, fid_ord):
        f_raw = float(_squeeze_B1(fidelity_idx)[0].item())
        levels = [float(l) for l in self.fid_levels]
        for lv in levels:
            if abs(f_raw - lv) < 1e-6: return lv
        if 0.0 <= f_raw < len(levels) and abs(f_raw - round(f_raw)) < 1e-6:
            return levels[int(round(f_raw))]
        return f_raw

    def _get_shared_coeffs(self, h, lecs):
        """Calculates the Affine Coefficients (shared) but NOT the xi (now split)."""
        B = h.size(0)
        phi = torch.cat([h, lecs], dim=-1)

        a_anchor = self.anchor_coeff_net(phi)
        a_ref_flat = self.refine_coeff_net(phi)
        a_ref = a_ref_flat.view(B, self.poly_order, self.nf + 1)

        return a_anchor, a_ref

    def _compute_xi_stack(self, modulator, fid_val, h, lecs):
        """Helper to compute polynomial stack of xi for a specific modulator."""
        xi = modulator(fid_val, self.f_min, h, lecs) # [B, 1]
        xi_p = [xi.pow(p+1) for p in range(self.poly_order)]
        return torch.stack(xi_p, dim=1) # [B, Poly, 1]

    def _primary_M(self, h, fid_ord, fid_val, lecs, a_anchor, a_ref):
        B = h.size(0)
        device = h.device
        k = int(fid_ord[0].item())
        m_k = self.matrix_sizes[k]
        xi_stack_H = self._compute_xi_stack(self.convergence_modulator_H, fid_val, h, lecs)

        Ik = self.I_full[:m_k, :m_k].to(device)
        Ms = a_anchor[:, :1].view(B, 1, 1) * Ik
        P_base_k = self._symmetrize(self.P_base)[:, :m_k, :m_k]
        Ms = Ms + torch.einsum("bi,imn->bmn", a_anchor[:, 1:], P_base_k)

        ref_id_coeffs = (a_ref[:, :, :1] * xi_stack_H).sum(dim=1)
        Ms = Ms + ref_id_coeffs.view(B, 1, 1) * Ik

        P_ref_k = self._symmetrize(self.P_refine)[:, :, :m_k, :m_k]
        a_ref_rest = a_ref[:, :, 1:]

        w_ref = a_ref_rest * xi_stack_H

        Ms_corr = torch.einsum("bpi,pimn->bmn", w_ref, P_ref_k)
        Ms = Ms + Ms_corr + self.eps_jitter * torch.eye(m_k, device=device).unsqueeze(0)

        return Ms, m_k

    def _secondary_S_mats(self, a_anchor, a_ref, xi_stack_obs_all, m_k):
        """
        xi_stack_obs_all: [B, NumObs, Poly, 1] - distinct flow per observable
        """
        if self.num_observables == 0: return None, None
        B = a_anchor.size(0)

        Ak_base = torch.stack([adapter(a_anchor) for adapter in self.obs_adapters], dim=1)

        poly_dim = a_ref.size(1)
        a_ref_flat = a_ref.view(B * poly_dim, -1)
        Ak_ref_list = []
        for adapter in self.obs_adapters:
            out_flat = adapter(a_ref_flat)
            Ak_ref_list.append(out_flat.view(B, poly_dim, -1))
        Ak_ref = torch.stack(Ak_ref_list, dim=1)

        H_base_k = self._symmetrize(self.H_base)[:, :, :m_k, :m_k]
        S_herm = torch.einsum("bql,qlij->bqij", Ak_base, H_base_k)

        H_ref_k = self._symmetrize(self.H_refine)[:, :, :, :m_k, :m_k]

        # Ak_ref: [B, Q, P, L] * xi_stack_obs: [B, Q, P, 1] -> [B, Q, P, L]
        C_eff = Ak_ref * xi_stack_obs_all

        S_herm = S_herm + torch.einsum("bqpl,pqlmn->bqmn", C_eff, H_ref_k)

        Q_base_k = self.Q_base[:, :, :m_k, :m_k]
        K = torch.einsum("bql,qlij->bqij", Ak_base, Q_base_k)

        Q_ref_k = self.Q_refine[:, :, :, :m_k, :m_k]
        K_ref = torch.einsum("bqpl,pqlmn->bqmn", C_eff, Q_ref_k)

        K_total = K + K_ref
        S_psd = torch.einsum("bqji,bqjk->bqik", K_total, K_total)

        return S_herm, S_psd

    def forward(
        self,
        z: torch.Tensor,
        n: torch.Tensor,
        fidelity_idx: torch.Tensor,
        lecs: torch.Tensor,
        shell: Optional[torch.Tensor] = None,
        trunc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z, n, fidelity_idx, lecs = map(_squeeze_B1, [z, n, fidelity_idx, lecs])

        h, fid_ord = self.backbone.latent(z, n, fidelity_idx, shell, trunc)
        fid_val = self._infer_fid_value(fidelity_idx, fid_ord)

        a_anchor, a_ref = self._get_shared_coeffs(h, lecs)

        M, m_k = self._primary_M(h, fid_ord, fid_val, lecs, a_anchor, a_ref)

        with torch.autocast(device_type=M.device.type if hasattr(M.device, "type") else "cpu", enabled=False):
            evals, V = torch.linalg.eigh(M.to(torch.float32))
        evals, V = evals.to(M.dtype), V.to(M.dtype)

        if self.num_observables > 0:
            xi_list = []
            for mod in self.convergence_modulator_obs:
                xi_list.append(self._compute_xi_stack(mod, fid_val, h, lecs))
            xi_stack_obs_all = torch.stack(xi_list, dim=1)
        else:
            xi_stack_obs_all = None

        S_herm, S_psd = self._secondary_S_mats(a_anchor, a_ref, xi_stack_obs_all, m_k)

        outs = []
        obs_cursor = 0
        for spec in self.output_specs:
            if spec["type"] == "energy":
                lvl = max(0, min(int(spec["level"]), evals.size(1) - 1))
                outs.append(evals[:, lvl])
            else:
                lvl = int(spec["level"])
                Sb = S_psd[:, obs_cursor] if bool(spec.get("psd", True)) else S_herm[:, obs_cursor]

                if self.use_soft_projector:
                    VS = torch.einsum("bmi,bij->bmj", V, Sb)
                    vSv = torch.einsum("bmi,bmi->bm", V, VS)
                    idxs = torch.arange(m_k, device=V.device)
                    w = torch.softmax(-((idxs - float(lvl)) ** 2) / max(self.soft_proj_tau, 1e-8), dim=0)
                    val = torch.einsum("bm,m->b", vSv, w.to(vSv.dtype))
                else:
                    v = V[..., lvl]
                    val = torch.einsum("bi,bij,bj->b", v, Sb, v)

                outs.append(val)
                obs_cursor += 1

        return torch.stack(outs, dim=1) + self.bias

    def regularization_loss(self) -> torch.Tensor:
        device = self.I_full.device
        reg = torch.zeros((), device=device)

        lam_l2 = float(getattr(self.config, "basis_l2_lambda", 0.0))
        if lam_l2 > 0:
            w = self.outer_weight
            reg = reg + lam_l2 * (self.P_base.pow(2) * w).sum()
            if self.H_base is not None:
                reg = reg + lam_l2 * (self.H_base.pow(2) * w).sum()
            if self.Q_base is not None:
                reg = reg + lam_l2 * (self.Q_base.pow(2) * w).sum()

            reg = reg + lam_l2 * (self.P_refine.pow(2)).sum()
            if self.H_refine is not None:
                reg = reg + lam_l2 * (self.H_refine.pow(2)).sum()
            if self.Q_refine is not None:
                reg = reg + lam_l2 * (self.Q_refine.pow(2)).sum()

        lam_sym = float(getattr(self.config, "symmetry_lambda", 0.0))
        if lam_sym > 0:
            P = self.P_base
            reg = reg + lam_sym * (P - P.transpose(-1, -2)).pow(2).sum()
            P_ref = self.P_refine
            reg = reg + lam_sym * (P_ref - P_ref.transpose(-1, -2)).pow(2).sum()

        return reg

    @torch.no_grad()
    def _pairwise_dist2(self, X: torch.Tensor) -> torch.Tensor:
        x2 = (X * X).sum(dim=1, keepdim=True)
        dist2 = x2 + x2.t() - 2.0 * (X @ X.t())
        return torch.clamp(dist2, min=0.0)

    def eigen_overlap_penalty(
        self,
        z, n, fidelity_idx, lecs,
        k: int = 2, sigma: float = 1.2,
        shell: Optional[torch.Tensor] = None,
        trunc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k <= 0: return torch.tensor(0.0, device=lecs.device)
        z, n, fidelity_idx, lecs = map(_squeeze_B1, [z, n, fidelity_idx, lecs])
        B = lecs.size(0)
        if B < 2: return torch.tensor(0.0, device=lecs.device)

        h, fid_ord = self.backbone.latent(z, n, fidelity_idx, shell, trunc)
        fid_val = self._infer_fid_value(fidelity_idx, fid_ord)
        a_anchor, a_ref = self._get_shared_coeffs(h, lecs)

        M, m_k = self._primary_M(h, fid_ord, fid_val, lecs, a_anchor, a_ref)

        with torch.autocast(device_type=M.device.type if hasattr(M.device, "type") else "cpu", enabled=False):
            _, V = torch.linalg.eigh(M.to(torch.float32))
        V = V.to(M.dtype)

        levels = sorted({int(s["level"]) for s in self.output_specs})
        ms = V.size(-1)
        levels = [max(0, min(L, ms - 1)) for L in levels]

        with torch.no_grad():
            D2 = self._pairwise_dist2(lecs.float())
            D2.fill_diagonal_(float("inf"))
            k_eff = min(k, B - 1)
            knn_idx = torch.topk(-D2, k=k_eff, dim=1).indices
            W = torch.exp(-torch.gather(D2, 1, knn_idx) / (2.0 * (sigma**2)))

        penalty = 0.0
        for L in levels:
            v = V[..., L]
            overlaps = (v.unsqueeze(1) * v[knn_idx]).sum(-1).abs() ** 2
            penalty += (W * (1.0 - overlaps)).mean()

        return penalty / max(1, len(levels))