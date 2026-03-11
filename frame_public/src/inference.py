import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Union, Optional, Mapping, Sequence, Any
from .config import PnBConfig
from .model import ParametricMatrixModelPnB
from .utils import load_object, ensure_shell_column, _load_model_state_dict


class InferenceHandler:
    def __init__(
        self,
        model_path: str,
        config: PnBConfig,
        fidelity_map: Dict[int, int],
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.fidelity_map = fidelity_map
        self.device = device or getattr(config, "device", torch.device("cpu"))

        self.model = ParametricMatrixModelPnB(config, list(fidelity_map.keys()))
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        state_dict = _load_model_state_dict(model_path, self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.scaler_X = load_object(config.scaler_X_path)
        self.scaler_y = load_object(config.scaler_y_path)

        self._use_shell = bool(getattr(self.config, "use_shell_embeddings", False))
        self._use_trunc = bool(getattr(self.config, "use_truncation_embeddings", False))
        self._trunc_vocab = int(getattr(self.config, "truncation_vocab_size", 0))

    def _prepare_optional_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if self._use_shell:
            out = ensure_shell_column(out, self.config.shell_vocab_size, prefer_by="N")

        if self._use_trunc and self._trunc_vocab > 0:
            if "trunc_id" not in out.columns:
                out["trunc_id"] = 0
            out["trunc_id"] = (
                out["trunc_id"].astype(int).clip(lower=0, upper=self._trunc_vocab - 1)
            )

        return out

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, only_mu: bool = False) -> np.ndarray:
        df = self._prepare_optional_columns(df)

        n = len(df)
        y_scaled_full = np.empty((n, self.config.num_outputs), dtype=np.float32)

        fid_vals_arr = df[self.config.fidelity_col].astype(int).to_numpy()
        fids_in_order = list(dict.fromkeys(fid_vals_arr))

        for fid_val in fids_in_order:
            idx = np.nonzero(fid_vals_arr == fid_val)[0]
            if idx.size == 0:
                continue

            sub = df.iloc[idx]
            X_scaled = self.scaler_X.transform(sub[self.config.input_cols])

            Z = torch.tensor(sub["Z"].values, dtype=torch.long, device=self.device)
            N = torch.tensor(sub["N"].values, dtype=torch.long, device=self.device)
            lecs = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            fid_physical = int(fid_val)
            fid = torch.full((len(sub),), fid_physical, dtype=torch.long, device=self.device)

            kwargs = {}
            if self._use_shell:
                kwargs["shell"] = torch.tensor(sub["shell_id"].values, dtype=torch.long, device=self.device)
            if self._use_trunc and self._trunc_vocab > 0:
                kwargs["trunc"] = torch.tensor(sub["trunc_id"].values, dtype=torch.long, device=self.device)

            y_scaled = self.model(z=Z, n=N, fidelity_idx=fid, lecs=lecs, **kwargs)
            y_scaled_full[idx] = y_scaled.detach().cpu().numpy()

        y_orig = self.scaler_y.inverse_transform(y_scaled_full)

        if "M1" in self.config.target_cols and "M1_2BC" in self.config.target_cols:
            J = None
            if "Jval" in df.columns:
                J = df["Jval"].values
            else:
                try:
                    J = self.try_infer_j_for_ca(df)["Jval"].values
                except (ValueError, KeyError):
                    J = None
            if J is not None:
                m1_idx = self.config.target_cols.index("M1")
                mu_1 = self.m1_to_mu(y_orig[:, m1_idx], J=J)
                m1_2bc_idx = self.config.target_cols.index("M1_2BC")
                mu_2bc = self.m1_to_mu(y_orig[:, m1_2bc_idx], J=J)
                mu = mu_1 + mu_2bc
                if only_mu:
                    return mu.reshape(-1, 1)
                y_orig = np.hstack(
                    (
                        y_orig,
                        mu_1.reshape(-1, 1),
                        mu_2bc.reshape(-1, 1),
                        mu.reshape(-1, 1),
                    )
                )

        if "E2" in self.config.target_cols:
            J = None
            if "Jval" in df.columns:
                J = df["Jval"].values
            else:
                try:
                    J = self.try_infer_j_for_ca(df)["Jval"].values
                except (ValueError, KeyError):
                    J = None
            if J is not None:
                e2_idx = self.config.target_cols.index("E2")
                q_1 = self.e2_to_q(y_orig[:, e2_idx], J=J)
                y_orig = np.hstack(
                    (
                        y_orig,
                        q_1.reshape(-1, 1),
                    )
                )

        return y_orig

    @staticmethod
    def m1_to_mu(m, J):
        # Correction factor to observable
        # Add =SQRT(4*PI()*B3/(3*(B3+1)*(2*B3+1)))
        factor = np.sqrt(4 * np.pi * J / (3 * (J + 1) * (2 * J + 1)))
        return factor * m

    @staticmethod
    def e2_to_q(e2, J):
        # Conversion factor from E2 to Q
        # Add =SQRT(16*PI()/5)*SQRT((B3)*(2*B3-1)/((B3+1)*(2*B3+1)*(2*B3+3)) )
        factor = np.sqrt(16 * np.pi / 5) * np.sqrt((J * (2 * J - 1)) / ((J + 1) * (2 * J + 1) * (2 * J + 3)))
        return factor * e2

    def transform_original_df_to_physical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds mu and Q columns to the original dataframe based on M1 and E2 predictions.
        Assumes df has columns named exactly as in config.target_cols (e.g. 'M1', 'M1_2BC', 'E2')
        plus a 'Jval' column.
        """
        df_out = df.copy()

        if "Jval" not in df_out.columns:
            # raise ValueError("Input dataframe must contain 'Jval' column to convert M1 to mu and E2 to Q")
            # Try to infer J values for odd-A Ca isotopes
            df_out = self.try_infer_j_for_ca(df_out)


        # ---- M1 -> mu ----
        if "M1" in self.config.target_cols:
            if "M1" not in df_out.columns or "M1_2BC" not in df_out.columns:
                raise KeyError("Dataframe must contain 'M1' and 'M1_2BC' columns.")

            mu_1 = self.m1_to_mu(df_out["M1"].to_numpy(), J=df_out["Jval"].to_numpy())
            mu_2bc = self.m1_to_mu(df_out["M1_2BC"].to_numpy(), J=df_out["Jval"].to_numpy())

            df_out["mu1bc"] = mu_1
            df_out["mu2bc"] = mu_2bc
            df_out["mu"] = mu_1 + mu_2bc

        # ---- E2 -> Q ----
        if "E2" in self.config.target_cols:
            if "E2" not in df_out.columns:
                raise KeyError("Dataframe must contain 'E2' column.")
            q_1 = self.e2_to_q(df_out["E2"].to_numpy(), J=df_out["Jval"].to_numpy())
            df_out["q"] = q_1

        return df_out

    def try_infer_j_for_ca(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tries to infer J values for odd-A Ca isotopes based on N and Z.
        Adds a 'Jval' column to the dataframe.
        """
        df_out = df.copy()

        def get_jval(N, Z):
            if Z == 20:
                if N in [17, 19]: return 3/2
                if N in [21, 23, 25, 27]: return 7/2
                if N in [29, 31]: return 3/2
                if N in [33]: return 1/2
                if N in [35]: return 5/2
            return np.nan  # or some default value / raise error

        df_out["Jval"] = df_out.apply(lambda row: get_jval(row["N"], row["Z"]), axis=1)

        if df_out["Jval"].isnull().any():
            raise ValueError("Some J values could not be inferred. Please check the input data.")

        return df_out

