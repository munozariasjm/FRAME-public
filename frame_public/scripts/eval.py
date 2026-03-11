import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

path2here = os.path.dirname(os.path.abspath(__file__))
path2src = os.path.abspath(os.path.join(path2here, ".."))
if path2src not in sys.path:
    sys.path.insert(0, path2src)

from src.config import PnBConfig
from src.inference import InferenceHandler

_Z2SYM = {
    1:"H", 2:"He", 3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne",
    11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",
    21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",
    31:"Ga",32:"Ge",33:"As",34:"Se",35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",
    41:"Nb",42:"Mo",43:"Tc",44:"Ru",45:"Rh",46:"Pd",
}

def isotope_token(z: int, n: int) -> str:
    A = int(z) + int(n)
    sym = _Z2SYM.get(int(z), f"Z{int(z)}")
    return f"{sym}-{A}"

def load_json(path: pathlib.Path):
    with open(path, "r") as f:
        return json.load(f)

def _rebuild_config_from_json(cfg_dict: Dict) -> PnBConfig:
    try:
        return PnBConfig(**cfg_dict)
    except Exception:
        cfg = PnBConfig()
        for k, v in cfg_dict.items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
        return cfg

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))

def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-12, None)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)

def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / np.clip(ss_tot, 1e-12, None))

def spearman_corr(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    def ranks(a):
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, ranks)
        avg = sums / counts
        return avg[inv]
    if len(x) < 3:
        return 0.0, 1.0
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = (np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()))
    if denom <= 0:
        return 0.0, 1.0
    rho = float((rx * ry).sum() / denom)
    from math import erf, sqrt
    t = rho * np.sqrt((len(x) - 2) / max(1e-12, 1 - rho**2))
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2))))
    return rho, float(p)

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def eta_squared(residuals: np.ndarray, groups: np.ndarray) -> float:
    y = np.asarray(residuals, dtype=float)
    g = np.asarray(groups)
    overall_var = np.var(y)
    if overall_var <= 0:
        return 0.0
    df = pd.DataFrame({"y": y, "g": g})
    mu_g = df.groupby("g")["y"].mean()
    counts = df.groupby("g")["y"].size().reindex(mu_g.index)
    explained = float(np.sum(counts.values * (mu_g.values - y.mean())**2) / len(y))
    return float(explained / overall_var)

def pretty_name_and_unit(name: str) -> Tuple[str, Optional[str]]:
    nl = name.lower()
    if ("energy" in nl) or nl.startswith("en"):
        return r"$E_B$", "MeV"
    if any(k in nl for k in ["rch","r_ch","charge radius","radius","radii","rch"]):
        return r"$R_{ch}$", "fm"
    if nl.startswith("m1"):
        return r"$M1$", None
    if nl.startswith("e2"):
        return r"$E2$", None
    return name, None

def axis_label(ref_or_emul: str, name: str) -> str:
    sym, unit = pretty_name_and_unit(name)
    if unit:
        return f"{ref_or_emul} {sym} ({unit})"
    return f"{ref_or_emul} {sym}"

_MARKERS = ["o","s","^","D","P","X","v","<",">","H","*","p","h","8"]

def parity_with_residual_panel(
    df: pd.DataFrame,
    y_col: str,
    yhat_col: str,
    color_key: str,
    marker_key: Optional[str],
    out_path: pathlib.Path,
):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.2, 8.2), dpi=180,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        sharex=True
    )

    cats = sorted(df[color_key].unique().tolist())
    cmap = plt.get_cmap("tab20")
    color_map = {c: cmap(i % 20) for i, c in enumerate(cats)}

    marker_map = None
    if marker_key is not None and marker_key in df.columns:
        ms = sorted(df[marker_key].astype(int).unique().tolist())
        marker_map = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(ms)}

    for _, row in df.iterrows():
        c = color_map[row[color_key]]
        m = 'o'
        if marker_map is not None:
            m = marker_map[int(row[marker_key])]
        ax_top.scatter(row[y_col], row[yhat_col], marker=m, alpha=0.9, s=26, c=[c], linewidths=0)
        ax_bot.scatter(row[y_col], row[yhat_col] - row[y_col], marker=m, alpha=0.9, s=20, c=[c], linewidths=0)

    lo = min(df[y_col].min(), df[yhat_col].min())
    hi = max(df[y_col].max(), df[yhat_col].max())
    ax_top.plot([lo, hi], [lo, hi], lw=1.2, ls="--", color="k")

    ax_bot.axhline(0.0, lw=1.0, color="k")

    ax_top.set_ylabel(axis_label("Emulated", y_col), fontsize=13)
    ax_bot.set_xlabel(axis_label("IMSRG", y_col), fontsize=13)
    sym, unit = pretty_name_and_unit(y_col)
    res_lab = rf"$\Delta$({sym})"
    if unit:
        res_lab += f" ({unit})"
    ax_bot.set_ylabel(res_lab, fontsize=12)

    legend_title = "Isotope" if color_key == "isotope" else color_key
    iso_handles = [Line2D([0], [0], marker='o', lw=0, label=str(c), color=color_map[c], markersize=6) for c in cats]
    leg1 = ax_top.legend(handles=iso_handles, title=legend_title, loc="upper left",
                         frameon=True, framealpha=0.85, fontsize=9, borderaxespad=0.6)

    if marker_map is not None:
        m_handles = [Line2D([0],[0], marker=mk, lw=0, color='k', label=f"t={tid}", markersize=7)
                     for tid, mk in sorted(marker_map.items())]
        leg2 = ax_top.legend(handles=m_handles, title="Truncation", loc="upper right",
                             frameon=True, framealpha=0.85, fontsize=9, borderaxespad=0.6)
        ax_top.add_artist(leg1)
    ax_top.tick_params(axis='both', which='major', labelsize=11)
    ax_bot.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def residual_hist(df: pd.DataFrame, res_col: str, out_path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=180)
    vals = df[res_col].values
    ax.hist(vals, bins=40, alpha=0.9)
    ax.axvline(0, color='k', lw=1)
    sym, unit = pretty_name_and_unit(res_col.replace("res_",""))
    base = rf"$\mathrm{{residual}}({sym})$"
    if unit:
        base += f" ({unit})"
    ax.set_xlabel(base, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def residual_heatmap_NxFid(
    df: pd.DataFrame,
    res_col: str,
    n_col: str,
    fid_col: str,
    out_path: pathlib.Path,
):
    if not {n_col, fid_col}.issubset(df.columns):
        return
    pivot = df.assign(abs_res=np.abs(df[res_col])).pivot_table(index=fid_col, columns=n_col, values="abs_res", aggfunc="mean")
    if pivot.empty:
        return
    data = pivot.values
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    im = ax.imshow(data, aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax, label=rf"mean |residual({pretty_name_and_unit(res_col.replace('res_',''))[0]})|")
    ax.set_yticks(np.arange(len(pivot.index)), labels=[str(v) for v in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[str(v) for v in pivot.columns])
    ax.set_xlabel("Neutron number $N$")
    ax.set_ylabel("Fidelity (emax)")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def residual_heatmap_NxTrunc(df: pd.DataFrame, res_col: str, out_path: pathlib.Path):
    if not {"N", "truncation"}.issubset(df.columns):
        return
    pivot = df.assign(abs_res=np.abs(df[res_col])).pivot_table(index="truncation", columns="N", values="abs_res", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=180)
    im = ax.imshow(pivot.values, aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax, label=rf"mean |residual({pretty_name_and_unit(res_col.replace('res_',''))[0]})|")
    ax.set_yticks(range(len(pivot.index)), [str(int(v)) for v in pivot.index])
    ax.set_xticks(range(len(pivot.columns)), [str(int(v)) for v in pivot.columns])
    ax.set_xlabel("Neutron number $N$")
    ax.set_ylabel("Truncation id")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def line_trends_by_isotope_over_N(df: pd.DataFrame, res_col: str, out_path: pathlib.Path, top_k: int = 16):
    if not {"Z","N"}.issubset(df.columns):
        return
    counts = df.groupby(["Z","N"]).size().sort_values(ascending=False)
    sel_isos = counts.index.tolist()[:top_k]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=180)
    handles = []; labels=[]
    for i, (Z,N) in enumerate(sel_isos):
        sub = df[(df["Z"]==Z) & (df["N"]==N)]
        if sub.empty: continue
        g = sub.groupby("N")[res_col].mean().sort_index()
        ax.plot(g.index.astype(int), g.values, marker="o", lw=1.8, color=cmap(i%20), alpha=0.95)
        A = int(Z)+int(N)
        handles.append(Line2D([0],[0], color=cmap(i%20), lw=2))
        labels.append(f"{_Z2SYM.get(int(Z),'Z'+str(int(Z)))}-{A}")
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel("Neutron number $N$")
    sym = pretty_name_and_unit(res_col.replace("res_",""))[0]
    ax.set_ylabel(rf"mean residual({sym})")
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.98), loc="upper left", frameon=False, fontsize=9, title="Isotope")
        plt.tight_layout(rect=[0,0,0.82,1])
    else:
        plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def line_trends_by_trunc_over_N_abs(df: pd.DataFrame, res_col: str, out_path: pathlib.Path, max_trunc_lines: int = 12):
    if not {"N","truncation"}.issubset(df.columns):
        return
    truncs = sorted(df["truncation"].astype(int).unique().tolist())[:max_trunc_lines]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=180)
    handles = []; labels=[]
    for i, t in enumerate(truncs):
        sub = df[df["truncation"].astype(int)==t].copy()
        if sub.empty: continue
        g = sub.assign(abs_res=np.abs(sub[res_col])).groupby("N")["abs_res"].mean().sort_index()
        ax.plot(g.index.astype(int), g.values, marker="o", lw=1.8, color=cmap(i%20), alpha=0.95)
        handles.append(Line2D([0],[0], color=cmap(i%20), lw=2))
        labels.append(f"t={t}")
    ax.set_xlabel("Neutron number $N$")
    sym = pretty_name_and_unit(res_col.replace("res_",""))[0]
    ax.set_ylabel(rf"mean |residual({sym})|")
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.98), loc="upper left", frameon=False, title="Truncation")
        plt.tight_layout(rect=[0,0,0.82,1])
    else:
        plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def boxplot_abs_residual_by_trunc(df: pd.DataFrame, res_col: str, out_path: pathlib.Path, min_count: int = 5):
    if "truncation" not in df.columns:
        return
    grouped = df.assign(abs_res=np.abs(df[res_col])).groupby("truncation")["abs_res"]
    data, labels = [], []
    for t, series in grouped:
        vals = series.dropna().to_numpy()
        if len(vals) >= min_count:
            data.append(vals)
            labels.append(int(t))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=180)
    ax.boxplot(data, labels=[str(l) for l in labels], showfliers=False)
    ax.set_xlabel("Truncation id")
    sym = pretty_name_and_unit(res_col.replace("res_",""))[0]
    ax.set_ylabel(rf"|residual({sym})|")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def _select_energy_and_radius_names(config: PnBConfig, out_names: List[str]) -> Tuple[Optional[str], Optional[str]]:
    e_name, r_name = None, None
    if getattr(config, "output_specs", None):
        for s in config.output_specs:
            t = str(s.get("type","")).lower()
            if t == "energy" and e_name is None:
                e_name = s["name"]
            if t in {"radius","radii","rch","charge_radius"} and r_name is None:
                r_name = s["name"]
    if e_name is None:
        for n in out_names:
            nl = n.lower()
            if any(k in nl for k in ["e", "energy", "binding"]):
                e_name = n; break
    if r_name is None:
        for n in out_names:
            nl = n.lower()
            if any(k in nl for k in ["rch", "r_ch", "radius", "radii"]):
                r_name = n; break
    return e_name, r_name

def element_energy_radii_panel(
    df_pred: pd.DataFrame,
    config: PnBConfig,
    out_names: List[str],
    fid_value: Optional[int],
    fid_col: str,
    out_dir: pathlib.Path,
    panel_Z: Optional[int] = None,
):
    energy_name, radii_name = _select_energy_and_radius_names(config, out_names)
    if (energy_name is None) or (radii_name is None):
        print("[WARN] Could not determine energy/radius outputs. Skipping element panel.")
        return

    df = df_pred.copy()
    if fid_value is not None and fid_col in df.columns:
        df = df.loc[df[fid_col].astype(int) == int(fid_value)].copy()

    if "Z" not in df.columns or "N" not in df.columns:
        print("[WARN] Columns Z and/or N not found. Skipping element panel.")
        return

    if panel_Z is None:
        z_counts = df.groupby("Z").size().sort_values(ascending=False)
        if len(z_counts) == 0:
            print("[WARN] No rows to select an element for panel. Skipping.")
            return
        panel_Z = int(z_counts.index[0])

    df = df.loc[df["Z"].astype(int) == int(panel_Z)].copy()
    if len(df) == 0:
        print(f"[WARN] No rows for Z={panel_Z}. Skipping element panel.")
        return

    iso_groups = df.groupby(["Z","N"])
    isotopes = list(iso_groups.groups.keys())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(isotopes)-1)) for i in range(len(isotopes))]

    marker_map = {}
    if "truncation" in df.columns:
        truncs = sorted(df["truncation"].astype(int).unique().tolist())
        markers = _MARKERS
        marker_map = {t: markers[i % len(markers)] for i, t in enumerate(truncs)}

    fig, (ax_energy, ax_radii) = plt.subplots(2, 1, figsize=(8, 10), dpi=180, constrained_layout=False)
    energy_errors, radii_errors = [], []

    for ((Z_val, N_val), col) in zip(isotopes, colors):
        sub = iso_groups.get_group((Z_val, N_val)).copy()
        if sub.empty:
            continue
        if "truncation" in sub.columns:
            sub = sub.sort_values("truncation")

        yE_true = sub[energy_name].to_numpy(dtype=float)
        yE_pred = sub[f"pred_{energy_name}"].to_numpy(dtype=float)
        yR_true = sub[radii_name].to_numpy(dtype=float)
        yR_pred = sub[f"pred_{radii_name}"].to_numpy(dtype=float)
        energy_errors.append(yE_pred - yE_true)
        radii_errors.append(yR_pred - yR_true)

        yE_unc = sub.get(f"unc_{energy_name}", pd.Series(np.zeros_like(yE_pred))).to_numpy()
        yR_unc = sub.get(f"unc_{radii_name}", pd.Series(np.zeros_like(yR_pred))).to_numpy()

        A_val = int(Z_val) + int(N_val)
        isotope_label = fr"$^{{{A_val}}}${_Z2SYM.get(int(Z_val), 'Z'+str(int(Z_val)))}"

        for i in range(len(sub)):
            m = 'o'
            if "truncation" in sub.columns:
                m = marker_map.get(int(sub.iloc[i]["truncation"]), 'o')

            ax_energy.errorbar([yE_true[i]],[yE_pred[i]],
                               yerr=[yE_unc[i]] if np.isfinite(yE_unc[i]) else None,
                               fmt=m, alpha=0.6, label=isotope_label if i==0 else None, color=col)
            ax_radii.errorbar([yR_true[i]],[yR_pred[i]],
                              yerr=[yR_unc[i]] if np.isfinite(yR_unc[i]) else None,
                              fmt=m, alpha=0.6, label=isotope_label if i==0 else None, color=col)

    energy_rmse = np.sqrt(np.mean(np.concatenate(energy_errors)**2)) if energy_errors else np.nan
    radii_rmse  = np.sqrt(np.mean(np.concatenate(radii_errors)**2)) if radii_errors else np.nan

    e_lo, e_hi = min(df[energy_name].min(), df[f"pred_{energy_name}"].min()), max(df[energy_name].max(), df[f"pred_{energy_name}"].max())
    r_lo, r_hi = min(df[radii_name].min(), df[f"pred_{radii_name}"].min()), max(df[radii_name].max(), df[f"pred_{radii_name}"].max())
    ax_energy.plot([e_lo, e_hi], [e_lo, e_hi], 'k--', lw=1.0)
    ax_radii.plot([r_lo, r_hi], [r_lo, r_hi], 'k--', lw=1.0)

    ax_energy.set_xlabel(axis_label("IMSRG", energy_name), fontsize=14)
    ax_energy.set_ylabel(axis_label("Emulated", energy_name), fontsize=14)
    ax_energy.tick_params(axis='both', which='major', labelsize=12)
    ax_energy.text(0.01, 0.99, rf'RMSE: {energy_rmse:.2f}', transform=ax_energy.transAxes, va='top', ha='left',
                   bbox=dict(facecolor='white', alpha=0.8), fontsize=12)

    ax_radii.set_xlabel(axis_label("IMSRG", radii_name), fontsize=14)
    ax_radii.set_ylabel(axis_label("Emulated", radii_name), fontsize=14)
    ax_radii.tick_params(axis='both', which='major', labelsize=12)
    ax_radii.text(0.01, 0.99, rf'RMSE: {radii_rmse:.2f}', transform=ax_radii.transAxes, va='top', ha='left',
                  bbox=dict(facecolor='white', alpha=0.8), fontsize=12)

    handles, labels = ax_energy.get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.02), ncol=2,
                   fontsize=11, frameon=False, title="Isotope")

    if "truncation" in df.columns and len(marker_map)>0:
        trunc_handles = [Line2D([0],[0], marker=mk, lw=0, color='k', label=f"t={tid}", markersize=7)
                         for tid, mk in sorted(marker_map.items())]
        fig.legend(trunc_handles, [f"t={tid}" for tid in sorted(marker_map)], loc="center right",
                   bbox_to_anchor=(0.98, 0.5), ncol=1, fontsize=11, frameon=False, title="Truncation")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    out_path = out_dir / f"predictions_vs_true_energy_radii_Z{panel_Z}.pdf"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Element IMSRG vs Emulated panel for Z={panel_Z}: {out_path}")

def compute_all_metrics(df: pd.DataFrame, out_names: List[str], fid_col: str) -> Dict[str, pd.DataFrame]:
    rows_overall, rows_fid, rows_iso, rows_trunc = [], [], [], []
    has_trunc = "truncation" in df.columns
    has_iso = "isotope" in df.columns

    for name in out_names:
        y = df[name].values
        yp = df[f"pred_{name}"].values
        rows_overall.append({
            "output": name, "RMSE": rmse(y, yp), "MAE": mae(y, yp),
            "MAPE_%": mape(y, yp), "R2": r2(y, yp),
        })

        if fid_col in df.columns:
            for fval, sub in df.groupby(fid_col):
                yb, ypb = sub[name].values, sub[f"pred_{name}"].values
                rows_fid.append({
                    "output": name, fid_col: int(fval),
                    "RMSE": rmse(yb, ypb), "MAE": mae(yb, ypb),
                    "MAPE_%": mape(yb, ypb), "R2": r2(yb, ypb),
                })

        if has_iso:
            for iso, sub in df.groupby("isotope"):
                yb, ypb = sub[name].values, sub[f"pred_{name}"].values
                rows_iso.append({
                    "output": name, "isotope": iso,
                    "RMSE": rmse(yb, ypb), "MAE": mae(yb, ypb),
                    "MAPE_%": mape(yb, ypb), "R2": r2(yb, ypb),
                })

        if has_trunc:
            for tval, sub in df.groupby("truncation"):
                yb, ypb = sub[name].values, sub[f"pred_{name}"].values
                rows_trunc.append({
                    "output": name, "truncation": int(tval),
                    "RMSE": rmse(yb, ypb), "MAE": mae(yb, ypb),
                    "MAPE_%": mape(yb, ypb), "R2": r2(yb, ypb),
                })

    out = {
        "overall": pd.DataFrame(rows_overall),
        "by_fidelity": pd.DataFrame(rows_fid),
        "by_isotope": pd.DataFrame(rows_iso) if rows_iso else pd.DataFrame(columns=["output","isotope","RMSE","MAE","MAPE_%","R2"]),
        "by_truncation": pd.DataFrame(rows_trunc) if rows_trunc else pd.DataFrame(columns=["output","truncation","RMSE","MAE","MAPE_%","R2"]),
    }
    return out

def main():
    ap = argparse.ArgumentParser(description="Evaluate PnB/BANNANE model on test split.")
    ap.add_argument("--exp_dir", type=str, required=True, help="Run directory (…/__TAG)")
    ap.add_argument("--ckpt", type=str, default="", help="Override checkpoint path; defaults to ckpt_best.pt")
    ap.add_argument("--fid_for_plots", type=str, default="highest", choices=["highest","all","value"],
                    help="Which fidelity for plots. 'value' pairs with --fid_value.")
    ap.add_argument("--fid_value", type=int, default=None, help="Fidelity value when --fid_for_plots=value")
    ap.add_argument("--panel_Z", type=int, default=None,
                    help="Element Z to use for the energy/radii panel. If omitted, auto-pick Z with most rows.")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.exp_dir).absolute()
    art_dir = run_dir / "artifacts"
    eval_dir = art_dir / "eval"
    ensure_dir(eval_dir)

    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        print(f"[ERROR] config.json not found: {cfg_path}"); sys.exit(2)
    cfg_json = load_json(cfg_path)
    cfg = _rebuild_config_from_json(cfg_json)

    fid_json_path = run_dir / "fidelity_map.json"
    if not fid_json_path.is_file():
        print(f"[ERROR] Missing fidelity_map.json at {fid_json_path}"); sys.exit(2)
    fid_map = {int(k): int(v) for k, v in load_json(fid_json_path).items()}

    test_csv = art_dir / "test.csv"
    if not test_csv.is_file():
        print(f"[ERROR] test.csv not found: {test_csv}"); sys.exit(2)
    df_test = pd.read_csv(test_csv)

    if ("Z" in df_test.columns) and ("N" in df_test.columns):
        df_test["isotope"] = [isotope_token(int(z), int(n)) for z, n in zip(df_test["Z"].values, df_test["N"].values)]

    # ----- Checkpoint -----
    ckpt = pathlib.Path(args.ckpt) if args.ckpt else (run_dir / "checkpoints" / "ckpt_best.pt")
    if not ckpt.is_file():
        alt = run_dir / "checkpoints" / "ckpt_last.pt"
        print(f"[WARN] {ckpt} not found; trying {alt}")
        ckpt = alt
    if not ckpt.is_file():
        print(f"[ERROR] No checkpoint found under {run_dir}/checkpoints/"); sys.exit(2)

    # device = torch.device(getattr(cfg, "device", "cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf = InferenceHandler(
        model_path=str(ckpt),
        config=cfg,
        fidelity_map=fid_map,
        device=device,
    )

    preds = inf.predict(df_test.copy())

    y_unc_all = None
    if hasattr(inf, "predict_with_uncertainty"):
        try:
            preds2, y_unc_all = inf.predict_with_uncertainty(df_test.copy())
            if isinstance(preds2, np.ndarray):
                preds = preds2
        except Exception:
            y_unc_all = None

    out_names = [s["name"] for s in cfg.output_specs] if getattr(cfg, "output_specs", None) else list(cfg.target_cols)

    df_pred = df_test.copy()
    for i, name in enumerate(out_names):
        df_pred[f"pred_{name}"] = preds[:, i]
        df_pred[f"res_{name}"]  = df_pred[f"pred_{name}"] - df_pred[name]
        if y_unc_all is not None and y_unc_all.ndim == 2 and y_unc_all.shape[1] == preds.shape[1]:
            df_pred[f"unc_{name}"] = y_unc_all[:, i]

    df_pred.to_csv(eval_dir / "preds_test.csv", index=False)

    fid_col = getattr(cfg, "fidelity_col", "emax")
    metrics = compute_all_metrics(df_pred, out_names, fid_col)
    metrics["overall"].to_csv(eval_dir / "metrics_overall.csv", index=False)
    metrics["by_fidelity"].to_csv(eval_dir / "metrics_by_fidelity.csv", index=False)
    metrics["by_isotope"].to_csv(eval_dir / "metrics_by_isotope.csv", index=False)
    metrics["by_truncation"].to_csv(eval_dir / "metrics_by_truncation.csv", index=False)

    trunc_report_path = eval_dir / "unique_truncations.txt"
    if "truncation" in df_pred.columns:
        unique_truncs_all = sorted(df_pred["truncation"].astype(int).unique().tolist())
        print(f"[DEBUG] Unique truncation IDs (all test rows): {unique_truncs_all}")
        with open(trunc_report_path, "w") as f:
            f.write("Unique truncation IDs (all test rows):\n")
            f.write(", ".join(map(str, unique_truncs_all)) + "\n")
    else:
        print("[DEBUG] Column 'truncation' not found; truncation-aware plots disabled.")

    if args.fid_for_plots == "highest":
        fid_value = max(list(fid_map.keys()))
    elif args.fid_for_plots == "value":
        if args.fid_value is None:
            print("[ERROR] --fid_for_plots=value requires --fid_value"); sys.exit(2)
        fid_value = int(args.fid_value)
    else:
        fid_value = None  # 'all'

    df_plot = df_pred.copy()
    if fid_value is not None and fid_col in df_plot.columns:
        df_plot = df_plot.loc[df_plot[fid_col].astype(int) == int(fid_value)].copy()

    for name in out_names:
        y_col = name
        yhat_col = f"pred_{name}"
        res_col = f"res_{name}"

        parity_with_residual_panel(
            df=df_plot,
            y_col=y_col,
            yhat_col=yhat_col,
            color_key="isotope" if "isotope" in df_plot.columns else fid_col,
            marker_key=("truncation" if "truncation" in df_plot.columns else None),
            out_path=eval_dir / f"parity_panel_{name}{'_fid'+str(fid_value) if fid_value is not None else ''}.png",
        )
        residual_hist(
            df=df_plot,
            res_col=res_col,
            out_path=eval_dir / f"residual_hist_{name}{'_fid'+str(fid_value) if fid_value is not None else ''}.png",
        )

        # Heatmaps/trends:
        if "truncation" in df_pred.columns:
            residual_heatmap_NxTrunc(
                df=df_pred[[ "N","truncation", res_col ]].dropna(),
                res_col=res_col,
                out_path=eval_dir / f"heatmap_mean_abs_residual_Nxtrunc_{name}.png",
            )
            line_trends_by_trunc_over_N_abs(
                df=df_pred[[ "N","truncation", res_col ]].dropna(),
                res_col=res_col,
                out_path=eval_dir / f"trend_trunc_over_N_abs_{name}.png",
            )
            boxplot_abs_residual_by_trunc(
                df=df_pred[[ "truncation", res_col ]].dropna(),
                res_col=res_col,
                out_path=eval_dir / f"box_abs_residual_by_trunc_{name}.png",
            )
        else:
            residual_heatmap_NxFid(
                df=df_pred[[ "N", fid_col, res_col ]].dropna() if fid_col in df_pred.columns else df_pred,
                res_col=res_col, n_col="N", fid_col=fid_col if fid_col in df_pred.columns else "fidelity",
                out_path=eval_dir / f"heatmap_mean_abs_residual_Nxfid_{name}.png",
            )
            line_trends_by_isotope_over_N(
                df=df_pred[[ "Z","N", res_col ]].dropna() if {"Z","N",res_col}.issubset(df_pred.columns) else df_pred,
                res_col=res_col,
                out_path=eval_dir / f"trend_isotope_over_N_{name}.png",
            )

    e_name = None
    for n in out_names:
        if ("energy" in n.lower()) or n.lower().startswith("en"):
            e_name = n; break
    if e_name and "truncation" in df_pred.columns:
        res_col = f"res_{e_name}"
        stats_rows = []
        for tid, sub in df_pred.groupby("truncation"):
            r = sub[res_col].values
            stats_rows.append({
                "truncation": int(tid),
                "count": int(len(r)),
                "mean_residual": float(np.mean(r)),
                "std_residual": float(np.std(r, ddof=0)),
                "sem_residual": float(np.std(r, ddof=0) / np.sqrt(max(1, len(r)))),
                "rmse": float(np.sqrt(np.mean(r**2))),
                "mean_abs_residual": float(np.mean(np.abs(r))),
                "median_abs_residual": float(np.median(np.abs(r))),
            })
        df_e_trunc = pd.DataFrame(stats_rows).sort_values("truncation")
        df_e_trunc.to_csv(eval_dir / "energy_residuals_by_truncation.csv", index=False)

        trunc_vals = df_pred["truncation"].astype(float).values
        e_res = df_pred[res_col].values
        rho_res, p_res = spearman_corr(trunc_vals, e_res)
        rho_abs, p_abs = spearman_corr(trunc_vals, np.abs(e_res))
        pd.DataFrame([{
            "output": e_name,
            "rho(truncation, residual)": rho_res, "p_approx_residual": p_res,
            "rho(truncation, |residual|)": rho_abs, "p_approx_abs": p_abs,
        }]).to_csv(eval_dir / "energy_truncation_correlation.csv", index=False)

        fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=180)
        ax.bar(range(len(df_e_trunc)), df_e_trunc["mean_residual"].to_numpy(),
               yerr=df_e_trunc["sem_residual"].to_numpy(), capsize=4)
        ax.set_xticks(range(len(df_e_trunc)), [str(int(v)) for v in df_e_trunc["truncation"].tolist()])
        ax.set_xlabel("Truncation id")
        sym = pretty_name_and_unit(e_name)[0]
        ax.set_ylabel(rf"mean residual({sym})")
        plt.tight_layout()
        fig.savefig(eval_dir / "energy_residual_by_truncation_bar.png", bbox_inches='tight')
        plt.close(fig)

        print(f"[EVAL] energy residual vs truncation: rho={rho_res:.3f} (p≈{p_res:.3g}), |res| rho={rho_abs:.3f} (p≈{p_abs:.3g})")

    element_energy_radii_panel(
        df_pred=df_pred,
        config=cfg,
        out_names=out_names,
        fid_value=(max(list(fid_map.keys())) if args.fid_for_plots == "highest"
                   else (args.fid_value if args.fid_for_plots == "value" else None)),
        fid_col=fid_col,
        out_dir=eval_dir,
        panel_Z=args.panel_Z,
    )

    print(f"[EVAL] Done.\n"
          f"  - Predictions: {eval_dir/'preds_test.csv'}\n"
          f"  - Metrics: {eval_dir/'metrics_overall.csv'}, {eval_dir/'metrics_by_fidelity.csv'}, "
          f"{eval_dir/'metrics_by_isotope.csv'}, {eval_dir/'metrics_by_truncation.csv'}\n"
          f"  - Parity/residual panels: parity_panel_*.png\n"
          f"  - Element panel: {eval_dir}/predictions_vs_true_energy_radii_Z*.pdf\n"
          f"  - Residual heatmaps/trends saved under: {eval_dir}")

if __name__ == "__main__":
    main()
