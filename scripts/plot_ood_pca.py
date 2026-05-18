
# =========================
# FULL CHEMICALLY-ANNOTATED PCA OOD PLOTS
# SAVES + SHOWS CORRECT INDIVIDUAL GRAPHS
# NO XPSSetDataset NEEDED
# =========================
from scripts.detr_model import SpectrumDETR

from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


REPS = ["skipatom", "matscholar", "soap"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
CONF_THRESH = 0.6
MAX_TRAIN_POINTS_FOR_PLOT = 5000

OUT_DIR = Path("ood_pca_plots_with_subsets")
OUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# FILE HELPERS
# --------------------------------------------------
def first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return p
    return None

def get_combo_paths(rep, holdout):
    return {
        "train_pt": first_existing([
            f"dataset_{rep}_train_no_holdout_{holdout}.pt",
            f"dataset_{rep}_synth_train_no_holdout_{holdout}.pt",
            f"dataset_{rep}_train_noCBr_CBr.pt" if holdout == "CBr" else None,
            f"dataset_{rep}_synth_train_noCBr_CBr.pt" if holdout == "CBr" else None,
        ]),
        "ood_pt": first_existing([
            f"dataset_{rep}_test_only_holdout_{holdout}.pt",
            f"dataset_{rep}_synth_test_only_holdout_{holdout}.pt",
            f"dataset_{rep}_test_onlyCBr_CBr.pt" if holdout == "CBr" else None,
            f"dataset_{rep}_synth_test_onlyCBr_CBr.pt" if holdout == "CBr" else None,
        ]),
        "ckpt": first_existing([
            f"detr_{rep}_holdout_{holdout}.pt",
            f"detr_{rep}_synth_holdout_{holdout}.pt",
            f"detr_{rep}_synth_train_noCBr_CBr.pt" if holdout == "CBr" else None,
        ]),
        "train_csv": first_existing([
            f"env_targets_{rep}_train_no_holdout_{holdout}.csv",
            f"env_targets_{rep}_synth_train_no_holdout_{holdout}.csv",
            f"env_targets_{rep}_train_noCBr_CBr.csv" if holdout == "CBr" else None,
        ]),
        "ood_csv": first_existing([
            f"env_targets_{rep}_test_only_holdout_{holdout}.csv",
            f"env_targets_{rep}_synth_test_only_holdout_{holdout}.csv",
            f"env_targets_{rep}_test_onlyCBr_CBr.csv" if holdout == "CBr" else None,
        ]),
    }

# --------------------------------------------------
# RAW PT HELPERS
# raw saved examples have keys:
#   key, spectrum, env_embeddings
# --------------------------------------------------
def load_raw_dataset(pt_path):
    return torch.load(pt_path, map_location="cpu")

def load_env_matrix_from_raw_pt(pt_path):
    ds = load_raw_dataset(pt_path)
    mats = []

    for i, ex in enumerate(ds):
        if not isinstance(ex, dict):
            raise TypeError(f"Example {i} in {pt_path} is not a dict: {type(ex)}")
        if "env_embeddings" not in ex:
            raise KeyError(f"Example {i} in {pt_path} missing 'env_embeddings'. Keys: {list(ex.keys())}")

        env = ex["env_embeddings"]
        if not torch.is_tensor(env):
            env = torch.tensor(env, dtype=torch.float32)

        env = env.detach().cpu().numpy()
        if env.ndim == 1:
            env = env[None, :]
        mats.append(env)

    return np.vstack(mats).astype(np.float32)

# --------------------------------------------------
# RAW DATALOADER HELPERS
# no XPSSetDataset required
# --------------------------------------------------
def collate_raw_examples(batch):
    spectra = []
    targets = []

    for i, ex in enumerate(batch):
        if not isinstance(ex, dict):
            raise TypeError(f"Batch example {i} is not a dict: {type(ex)}")
        if "spectrum" not in ex:
            raise KeyError(f"Batch example {i} missing 'spectrum'. Keys: {list(ex.keys())}")
        if "env_embeddings" not in ex:
            raise KeyError(f"Batch example {i} missing 'env_embeddings'. Keys: {list(ex.keys())}")

        x = ex["spectrum"]
        y = ex["env_embeddings"]

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)

        # spectrum -> [1, N]
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # env embeddings -> [n_env, d_env]
        if y.ndim == 1:
            y = y.unsqueeze(0)

        spectra.append(x.float())
        targets.append(y.float())

    spectra = torch.stack(spectra, dim=0)

    return {
        "spectra": spectra,
        "targets": targets,
    }

def make_loader(pt_path, batch_size=BATCH_SIZE):
    ds = load_raw_dataset(pt_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_raw_examples,
        num_workers=0
    )

def get_batch_spectra(batch):
    if not isinstance(batch, dict):
        raise TypeError(f"Expected batch dict, got {type(batch)}")
    if "spectra" not in batch:
        raise KeyError(f"Batch missing 'spectra'. Keys are: {list(batch.keys())}")
    return batch["spectra"]

# --------------------------------------------------
# MODEL HELPERS
# this matches the version you were using earlier
# --------------------------------------------------
def build_model():
    model = SpectrumDETR(
        d_model=256,
        d_env=128,
        n_queries=24,
        n_heads=8,
        n_layers=3,
    )
    return model

def load_model_from_ckpt(ckpt_path, device=DEVICE):
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model

# --------------------------------------------------
# PREDICTION EXTRACTION
# expected model output:
#   (env_preds, conf_logits)
# or
#   (env_preds, something, conf_logits)
# --------------------------------------------------
def extract_pred_env_vectors(model, loader, device, conf_thresh=CONF_THRESH):
    preds = []

    with torch.no_grad():
        for batch in loader:
            x = get_batch_spectra(batch).to(device)
            out = model(x)

            if not isinstance(out, (tuple, list)):
                raise TypeError(f"Model output should be tuple/list, got {type(out)}")

            if len(out) == 2:
                env_preds, conf_logits = out
            elif len(out) == 3:
                env_preds, _, conf_logits = out
            else:
                raise ValueError(f"Unexpected model output length: {len(out)}")

            conf = torch.sigmoid(conf_logits) if conf_logits.ndim == 2 else conf_logits
            keep_mask = conf > conf_thresh

            for b in range(env_preds.shape[0]):
                keep = keep_mask[b]
                if keep.sum() > 0:
                    preds.append(env_preds[b][keep].detach().cpu())

    if len(preds) == 0:
        return np.empty((0, 128), dtype=np.float32)

    return torch.cat(preds, dim=0).numpy().astype(np.float32)

# --------------------------------------------------
# CHEMISTRY HELPERS
# --------------------------------------------------
def mol_from_smiles(smi):
    if not isinstance(smi, str) or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi)

def get_atom(mol, idx):
    if mol is None:
        return None
    try:
        idx = int(idx)
    except Exception:
        return None
    if idx < 0 or idx >= mol.GetNumAtoms():
        return None
    return mol.GetAtomWithIdx(idx)

def is_C_halogen(smiles, atom_index, Z):
    mol = mol_from_smiles(smiles)
    atom = get_atom(mol, atom_index)
    if atom is None or atom.GetAtomicNum() != 6:
        return False
    return any(n.GetAtomicNum() == Z for n in atom.GetNeighbors())

def is_carbonyl_oxygen(smiles, atom_index):
    mol = mol_from_smiles(smiles)
    atom = get_atom(mol, atom_index)
    if atom is None or atom.GetAtomicNum() != 8:
        return False

    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() == 6 and bond.GetBondTypeAsDouble() == 2.0:
            return True
    return False

def is_other_oxygen(smiles, atom_index):
    mol = mol_from_smiles(smiles)
    atom = get_atom(mol, atom_index)
    if atom is None or atom.GetAtomicNum() != 8:
        return False
    return not is_carbonyl_oxygen(smiles, atom_index)

def is_aromatic_carbon(smiles, atom_index):
    mol = mol_from_smiles(smiles)
    atom = get_atom(mol, atom_index)
    if atom is None or atom.GetAtomicNum() != 6:
        return False
    return atom.GetIsAromatic()

def is_non_aromatic_carbon(smiles, atom_index):
    mol = mol_from_smiles(smiles)
    atom = get_atom(mol, atom_index)
    if atom is None or atom.GetAtomicNum() != 6:
        return False
    return not atom.GetIsAromatic()

# --------------------------------------------------
# NAME NORMALISATION
# --------------------------------------------------
def canonicalise_polymer_name(x):
    if pd.isna(x):
        return None

    s = str(x).strip().lower()
    s = re.sub(r"\([^)]*\)", "", s)

    if s.startswith("poly(") and s.endswith(")"):
        s = s[5:-1].strip()

    s = re.sub(r"^poly[-\s]+", "", s)
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s

# --------------------------------------------------
# CORRECTED METADATA LOADER
# --------------------------------------------------
def load_polymer_metadata(metadata_csv="polymer_matscholar_env_vectors.csv"):
    meta = pd.read_csv(metadata_csv)

    required = {"name", "smiles", "atom_index", "atom_symbol"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"{metadata_csv} missing required columns: {missing}")

    meta = meta.copy()
    meta["meta_name"] = meta["name"].astype(str).str.strip()
    meta["meta_norm_key"] = meta["meta_name"].map(canonicalise_polymer_name)
    meta["atom_index"] = pd.to_numeric(meta["atom_index"], errors="coerce")

    meta = meta[
        ["meta_norm_key", "smiles", "atom_index", "atom_symbol"]
    ]

    meta = meta.drop_duplicates(subset=["meta_norm_key", "atom_index"])
    return meta

META = load_polymer_metadata("polymer_matscholar_env_vectors.csv")

# --------------------------------------------------
# CORRECTED SUBSET MASK BUILDER
# --------------------------------------------------
def build_training_subset_masks(train_csv_path, holdout, meta_df=META):
    env_df = pd.read_csv(train_csv_path).copy()

    required = {"norm_key", "env_index_within_polymer"}
    missing = required - set(env_df.columns)
    if missing:
        raise ValueError(f"{train_csv_path} missing required columns: {missing}")

    env_df["norm_key"] = env_df["norm_key"].astype(str).str.strip().map(canonicalise_polymer_name)
    env_df["env_index_within_polymer"] = pd.to_numeric(env_df["env_index_within_polymer"], errors="coerce")

    merged = env_df.merge(
        meta_df,
        left_on=["norm_key", "env_index_within_polymer"],
        right_on=["meta_norm_key", "atom_index"],
        how="left"
    )

    unmatched = merged["smiles"].isna().sum()
    matched = len(merged) - unmatched
    print(f"Matched metadata rows: {matched}/{len(merged)}")

    smiles = merged["smiles"].values
    atom_idx = merged["atom_index"].values

    masks = {}

    if holdout == "CBr":
        masks["C–Cl"] = np.array([is_C_halogen(s, i, 17) for s, i in zip(smiles, atom_idx)])
        masks["C–F"]  = np.array([is_C_halogen(s, i, 9)  for s, i in zip(smiles, atom_idx)])

    elif holdout == "carbonyl_O":
        masks["other O"] = np.array([is_other_oxygen(s, i) for s, i in zip(smiles, atom_idx)])

    elif holdout == "aromatic_C":
        masks["other C"] = np.array([is_non_aromatic_carbon(s, i) for s, i in zip(smiles, atom_idx)])

    return merged, masks

# --------------------------------------------------
# OPTIONAL DEBUG HELPER
# --------------------------------------------------
def sanity_check_one(train_csv_path, holdout):
    merged, masks = build_training_subset_masks(train_csv_path, holdout)
    print(f"\nSanity check for {Path(train_csv_path).name} | {holdout}")
    print("rows:", len(merged))
    print("matched smiles:", int(merged["smiles"].notna().sum()))
    for k, m in masks.items():
        print(f"{k}: {int(m.sum())}")

# --------------------------------------------------
# SINGLE-PLOT FUNCTION
# saves + shows inline
# --------------------------------------------------
def plot_combo_with_subsets(
    rep,
    holdout,
    X_train_128,
    X_ood_true_128,
    X_pred_128,
    train_csv_path,
    out_dir=OUT_DIR,
    show_saved_image=False,
):
    X_train_bg = X_train_128
    if len(X_train_bg) > MAX_TRAIN_POINTS_FOR_PLOT:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_train_bg), size=MAX_TRAIN_POINTS_FOR_PLOT, replace=False)
        X_train_bg = X_train_bg[idx]

    scaler_plot = StandardScaler().fit(X_train_bg)
    X_train_bg_scaled = scaler_plot.transform(X_train_bg)

    pca_plot = PCA(n_components=2, random_state=0).fit(X_train_bg_scaled)

    Z_train_bg = pca_plot.transform(X_train_bg_scaled)
    Z_ood = pca_plot.transform(scaler_plot.transform(X_ood_true_128))
    Z_pred = pca_plot.transform(scaler_plot.transform(X_pred_128))
    Z_train_full = pca_plot.transform(scaler_plot.transform(X_train_128))

    merged_df, subset_masks = build_training_subset_masks(train_csv_path, holdout)

    if len(merged_df) != len(X_train_128):
        raise ValueError(
            f"Row mismatch for {rep} | {holdout}: "
            f"train CSV has {len(merged_df)} rows but train env matrix has {len(X_train_128)} rows."
        )

    Z_ref = np.vstack([Z_train_bg, Z_ood])
    x_lo, x_hi = np.percentile(Z_ref[:, 0], [1, 99])
    y_lo, y_hi = np.percentile(Z_ref[:, 1], [1, 99])

    pad_x = 0.10 * (x_hi - x_lo + 1e-12)
    pad_y = 0.10 * (y_hi - y_lo + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for ax in axes:
        ax.scatter(
            Z_train_bg[:, 0], Z_train_bg[:, 1],
            s=4, alpha=0.04, color="grey",
            label="Training envs"
        )

        for label, mask in subset_masks.items():
            if mask.sum() == 0:
                continue
            Z_sub = Z_train_full[mask]
            ax.scatter(
                Z_sub[:, 0], Z_sub[:, 1],
                s=30, alpha=0.9,
                color=SUBSET_COLORS.get(label, "blue"),
                label=label
            )

        ax.scatter(
            Z_ood[:, 0], Z_ood[:, 1],
            s=44, alpha=0.92, color="green",
            label=f"True OOD ({HOLDOUT_LABELS.get(holdout, holdout)})"
        )

        ax.scatter(
            Z_pred[:, 0], Z_pred[:, 1],
            s=48, alpha=0.90, color="red",
            label="Predicted OOD envs"
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    axes[0].set_title(f"{rep}: {holdout} OOD placement (full)")
    axes[1].set_title(f"{rep}: {holdout} OOD placement (zoomed)")
    axes[1].set_xlim(x_lo - pad_x, x_hi + pad_x)
    axes[1].set_ylim(y_lo - pad_y, y_hi + pad_y)
    axes[1].legend(frameon=False, fontsize=9)

    save_path = out_dir / f"ood_pca_{rep}_{holdout}_with_subsets.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {save_path}")
