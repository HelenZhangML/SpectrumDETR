"""
Dataset construction pipeline for polymer XPS environment prediction.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


# ============================================================
# CONFIG
# ============================================================

SPECTRA_DIR = Path("data/experimental_data")

REP_CFG = {
    "soap": dict(
        ENV_CSV=Path("soap_env_vectors.csv"),
        ENV_PREFIX="soap_env_",
    ),
    "skipatom": dict(
        ENV_CSV=Path("polymer_skipatom_env_vectors.csv"),
        ENV_PREFIX="env_",
    ),
    "matscholar": dict(
        ENV_CSV=Path("polymer_matscholar_env_vectors.csv"),
        ENV_PREFIX="matscholar_env_",
    ),
}

ALIASES = {
    "poly(2-chlolorostyrene)": "poly(2-chlorostyrene)",
}

HOLDOUT_NAMES = ["CBr", "carbonyl_O", "aromatic_C"]

N_POINTS = 1024
NORMALISE_MODE = "max"

USE_PCA = True
D_ENV = 128


# ============================================================
# KEY NORMALISATION
# ============================================================

def norm_key(s: str) -> str:
    s = str(s).strip()

    if s == ".DS_Store":
        return ""

    s = re.sub(r"\s+", " ", s.lower()).strip()

    while True:
        m = re.match(r"^(.*)\s+\(([^()]*)\)\s*$", s)
        if not m:
            break

        base = m.group(1).strip()
        tag = m.group(2).strip()

        tag_like = (
            len(tag) <= 20
            and re.fullmatch(r"[a-z0-9\- ]+", tag) is not None
        )

        base_poly_like = (
            "poly(" in base
            or base.startswith("nylon")
            or base in {
                "cellulose",
                "ethylcellulose",
                "hydroxypropylycellulose",
            }
        )

        if tag_like and base_poly_like:
            s = base
            continue

        break

    return ALIASES.get(s, s)


# ============================================================
# HOLDOUT RULES
# ============================================================

def atom_is_cbr(mol: Chem.Mol, atom_idx: int) -> bool:
    a = mol.GetAtomWithIdx(atom_idx)
    return (
        a.GetAtomicNum() == 6
        and any(n.GetAtomicNum() == 35 for n in a.GetNeighbors())
    )


def atom_is_carbonyl_oxygen(mol: Chem.Mol, atom_idx: int) -> bool:
    a = mol.GetAtomWithIdx(atom_idx)

    if a.GetAtomicNum() != 8:
        return False

    for b in a.GetBonds():
        other = b.GetOtherAtom(a)
        if (
            other.GetAtomicNum() == 6
            and b.GetBondType() == Chem.rdchem.BondType.DOUBLE
        ):
            return True

    return False


def atom_is_aromatic_carbon(mol: Chem.Mol, atom_idx: int) -> bool:
    a = mol.GetAtomWithIdx(atom_idx)
    return a.GetAtomicNum() == 6 and a.GetIsAromatic()


def atom_matches_holdout(
    mol: Chem.Mol,
    atom_idx: int,
    holdout_name: Optional[str],
) -> bool:
    if holdout_name == "CBr":
        return atom_is_cbr(mol, atom_idx)

    if holdout_name == "carbonyl_O":
        return atom_is_carbonyl_oxygen(mol, atom_idx)

    if holdout_name == "aromatic_C":
        return atom_is_aromatic_carbon(mol, atom_idx)

    return False


# ============================================================
# FEATURE STANDARDISATION + PCA
# ============================================================

def scaler_fit(X: np.ndarray) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    return {
        "mean": mean.squeeze(0).astype(np.float32),
        "std": std.squeeze(0).astype(np.float32),
    }


def scaler_transform(X: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)


def pca_fit(X_scaled: np.ndarray, d_out: int) -> Dict[str, np.ndarray]:
    mean = X_scaled.mean(axis=0, keepdims=True)
    Xc = X_scaled - mean

    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:d_out].T

    return {
        "mean": mean.squeeze(0).astype(np.float32),
        "W": W.astype(np.float32),
        "d_out": int(d_out),
    }


def pca_transform(X_scaled: np.ndarray, pca: Dict[str, np.ndarray]) -> np.ndarray:
    return ((X_scaled - pca["mean"]) @ pca["W"]).astype(np.float32)


# ============================================================
# SPECTRUM HELPERS
# ============================================================

def _normalise(y: np.ndarray, mode: str = "max") -> np.ndarray:
    y = y.astype(np.float32)

    if mode == "max":
        return y / (np.max(np.abs(y)) + 1e-12)

    if mode == "zscore":
        return ((y - y.mean()) / (y.std() + 1e-12)).astype(np.float32)

    return y


def _interp_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(x_grid, x[order], y[order]).astype(np.float32)


def load_spectrum_file(path: Path):
    """
    Load common two-column spectrum files.

    Expected formats:
        CSV/TXT/DAT with two numeric columns:
        binding energy, intensity
    """
    try:
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] >= 2:
            x = numeric.iloc[:, 0].to_numpy(dtype=np.float32)
            y = numeric.iloc[:, 1].to_numpy(dtype=np.float32)
            return x, y
    except Exception:
        pass

    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.loadtxt(path)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Could not read spectrum as two-column data: {path}")

    x = arr[:, 0].astype(np.float32)
    y = arr[:, 1].astype(np.float32)

    return x, y


def load_all_spectra(spectra_dir: Path):
    spectrum_paths = []

    for ext in ("*.csv", "*.txt", "*.dat", "*.tsv"):
        spectrum_paths.extend(spectra_dir.glob(ext))

    if not spectrum_paths:
        raise FileNotFoundError(f"No spectra files found in {spectra_dir}")

    spectra = {}

    for path in spectrum_paths:
        key = norm_key(path.stem)
        if not key:
            continue

        try:
            x, y = load_spectrum_file(path)
            spectra.setdefault(key, []).append({
                "path": str(path),
                "x": x,
                "y": y,
            })
        except Exception as exc:
            print(f"Skipping spectrum {path}: {exc}")

    return spectra


# ============================================================
# REPRESENTATION PREPROCESSING
# ============================================================

def load_and_preprocess_env_vectors(rep: str):
    cfg = REP_CFG[rep]

    env_df = pd.read_csv(cfg["ENV_CSV"])
    env_df.columns = [c.strip().lower() for c in env_df.columns]

    feat_cols = [
        c for c in env_df.columns
        if c.startswith(cfg["ENV_PREFIX"].lower())
    ]

    if not feat_cols:
        raise ValueError(
            f"No feature columns found for rep={rep} "
            f"with prefix={cfg['ENV_PREFIX']}"
        )

    print("Loaded env vectors:", len(env_df))
    print("Raw feature dim:", len(feat_cols))

    X_raw = env_df[feat_cols].to_numpy(dtype=np.float32)

    scaler = scaler_fit(X_raw)
    X_scaled = scaler_transform(X_raw, scaler)

    if USE_PCA:
        pca = pca_fit(X_scaled, D_ENV)
        X_env = pca_transform(X_scaled, pca)
    else:
        pca = None
        X_env = X_scaled

    print("Final env dim:", X_env.shape[1])

    env_cols = [f"env_pca_{i}" for i in range(X_env.shape[1])]

    env_processed = pd.concat(
        [
            env_df.drop(columns=feat_cols).reset_index(drop=True),
            pd.DataFrame(X_env, columns=env_cols),
        ],
        axis=1,
    )

    preprocess = {
        "scaler": {
            "mean": scaler["mean"].tolist(),
            "std": scaler["std"].tolist(),
        },
        "pca": None if pca is None else {
            "mean": pca["mean"].tolist(),
            "W": pca["W"].tolist(),
            "d_out": pca["d_out"],
        },
        "raw_feature_cols": feat_cols,
        "processed_feature_cols": env_cols,
        "use_pca": USE_PCA,
        "d_env": int(X_env.shape[1]),
    }

    return env_processed, preprocess


# ============================================================
# LABELLED TARGET EXPORT
# ============================================================

def holdout_label_for_atom(smiles: str, atom_idx: int) -> str:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return "unknown"

    labels = []

    for name in HOLDOUT_NAMES:
        if atom_matches_holdout(mol, atom_idx, name):
            labels.append(name)

    if labels:
        return ";".join(labels)

    return "other"


def make_labelled_targets(env_df: pd.DataFrame, env_cols: List[str]) -> pd.DataFrame:
    rows = []

    for _, row in env_df.iterrows():
        smiles = row["smiles"]
        atom_idx = int(row["atom_index"])

        rows.append({
            "name": row.get("name", ""),
            "norm_key": row.get("norm_key", ""),
            "smiles": smiles,
            "atom_index": atom_idx,
            "atom_symbol": row.get("atom_symbol", ""),
            "target_label": holdout_label_for_atom(smiles, atom_idx),
            **{c: row[c] for c in env_cols},
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_dataset(
    rep: str = "matscholar",
    split_mode: str = "all",
    holdout_name: Optional[str] = None,
):
    print("=" * 60)
    print("REP =", rep)
    print("SPLIT_MODE =", split_mode)
    print("HOLDOUT =", holdout_name)
    print("=" * 60)

    if rep not in REP_CFG:
        raise ValueError(f"Unknown rep: {rep}. Choose from {list(REP_CFG)}")

    if split_mode not in {"all", "train_no_holdout", "test_only_holdout"}:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if split_mode != "all" and holdout_name not in HOLDOUT_NAMES:
        raise ValueError(
            f"holdout_name must be one of {HOLDOUT_NAMES} for split_mode={split_mode}"
        )

    env_df, preprocess = load_and_preprocess_env_vectors(rep)

    if "name" not in env_df.columns:
        raise ValueError("Environment CSV must contain a 'name' column.")

    if "smiles" not in env_df.columns:
        raise ValueError("Environment CSV must contain a 'smiles' column.")

    if "atom_index" not in env_df.columns:
        raise ValueError("Environment CSV must contain an 'atom_index' column.")

    env_df["norm_key"] = env_df["name"].apply(norm_key)
    env_cols = preprocess["processed_feature_cols"]

    # Holdout filtering
    if split_mode in {"train_no_holdout", "test_only_holdout"}:
        keep_mask = []

        for _, row in env_df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])

            if mol is None:
                keep_mask.append(False)
                continue

            atom_idx = int(row["atom_index"])
            match = atom_matches_holdout(mol, atom_idx, holdout_name)

            if split_mode == "train_no_holdout":
                keep_mask.append(not match)
            else:
                keep_mask.append(match)

        env_df = env_df.loc[keep_mask].reset_index(drop=True)

    print("Environment rows after holdout filtering:", len(env_df))

    grouped_envs = {}

    for key, group in env_df.groupby("norm_key"):
        X = group[env_cols].to_numpy(dtype=np.float32)
        grouped_envs[key] = torch.tensor(X, dtype=torch.float32)

    print("Grouped polymers with environments:", len(grouped_envs))

    spectra = load_all_spectra(SPECTRA_DIR)

    all_x = np.concatenate([
        s["x"] for entries in spectra.values() for s in entries
    ])

    x_min = float(np.min(all_x))
    x_max = float(np.max(all_x))

    x_grid = np.linspace(x_min, x_max, N_POINTS).astype(np.float32)

    dataset = []
    missing_env = []

    for key, entries in spectra.items():
        if key not in grouped_envs:
            missing_env.append(key)
            continue

        for entry in entries:
            y_grid = _interp_to_grid(entry["x"], entry["y"], x_grid)
            y_grid = _normalise(y_grid, NORMALISE_MODE)

            dataset.append({
                "key": key,
                "source_path": entry["path"],
                "spectrum": torch.tensor(y_grid, dtype=torch.float32),
                "env_embeddings": grouped_envs[key].clone(),
            })

    if not dataset:
        raise RuntimeError(
            "No dataset entries were built. Check that spectrum filenames match "
            "environment CSV polymer names after norm_key()."
        )

    # Output paths
    if split_mode == "all":
        pt_path = Path(f"dataset_{rep}.pt")
        meta_path = Path(f"dataset_meta_{rep}.json")
        targets_path = Path(f"env_targets_{rep}.csv")
    else:
        suffix = f"_{split_mode}_{holdout_name}"
        pt_path = Path(f"dataset_{rep}{suffix}.pt")
        meta_path = Path(f"dataset_meta_{rep}{suffix}.json")
        targets_path = Path(f"env_targets_{rep}{suffix}.csv")

    torch.save(dataset, pt_path)

    labelled_targets = make_labelled_targets(env_df, env_cols)
    labelled_targets.to_csv(targets_path, index=False)

    meta = {
        "rep": rep,
        "split_mode": split_mode,
        "holdout_name": holdout_name,
        "n_examples": len(dataset),
        "n_polymers_with_envs": len(grouped_envs),
        "n_spectrum_keys": len(spectra),
        "missing_env_keys": sorted(set(missing_env)),
        "x_min": x_min,
        "x_max": x_max,
        "n_points": N_POINTS,
        "normalise_mode": NORMALISE_MODE,
        "env_dim": preprocess["d_env"],
        "preprocess": preprocess,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved dataset:", pt_path, "n=", len(dataset))
    print("Saved metadata:", meta_path)
    print("Saved labelled targets:", targets_path)

    return {
        "dataset": dataset,
        "meta": meta,
        "env_df": env_df,
        "grouped_envs": grouped_envs,
        "preprocess": preprocess,
    }


# ============================================================
# BUILD ALL
# ============================================================

def build_all(rep: str):
    build_dataset(rep=rep, split_mode="all", holdout_name=None)

    for holdout in HOLDOUT_NAMES:
        build_dataset(rep=rep, split_mode="train_no_holdout", holdout_name=holdout)
        build_dataset(rep=rep, split_mode="test_only_holdout", holdout_name=holdout)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    for rep_name in ["skipatom", "matscholar", "soap"]:
        build_all(rep_name)
