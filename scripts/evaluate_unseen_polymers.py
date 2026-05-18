"""
Evaluate DETR models on independently measured unseen polymer XPS spectra.

This script:
1. Parses experimental VGD C1s spectra.
2. Preprocesses spectra for model input.
3. Predicts environment embeddings using trained DETR checkpoints.
4. Matches predicted embeddings to nearest training environments.
5. Converts predictions to coarse chemical environment labels.
6. Compares predicted environments against structure-derived ground truth.
"""

from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from vgd_reader import read_vgd

from scripts.detr_model import SpectrumDETR


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENTAL_DIR = Path("Helen_Z_Polymers_unzipped/Helen Z Polymers")
TARGET_LEN = 6600
CONF_THRESH = 0.30

POLYMER_SMILES = {
    "pDMA Sample 1": "CCC(C(=O)N(C)C)C",
    "pHEMA Sample 2": "CCC(C(=O)OCCO)C",
    "pHPMA Sample 3": "CCC(C(=O)NCC(C)O)C",
}

HOLDOUT = "carbonyl_O"

REP_CONFIGS = {
    "SkipAtom": {
        "full": {
            "checkpoint": "detr_skipatom.pt",
            "train_pt": "dataset_skipatom.pt",
            "train_csv": "env_targets_skipatom.csv",
            "metadata_csv": "polymer_skipatom_env_vectors.csv",
        },
        "holdout": {
            "checkpoint": f"detr_skipatom_holdout_{HOLDOUT}.pt",
            "train_pt": f"dataset_skipatom_train_no_holdout_{HOLDOUT}.pt",
            "train_csv": f"env_targets_skipatom_train_no_holdout_{HOLDOUT}.csv",
            "metadata_csv": "polymer_skipatom_env_vectors.csv",
        },
    },
    "MatScholar": {
        "full": {
            "checkpoint": "detr_matscholar.pt",
            "train_pt": "dataset_matscholar.pt",
            "train_csv": "env_targets_matscholar.csv",
            "metadata_csv": "polymer_matscholar_env_vectors.csv",
        },
        "holdout": {
            "checkpoint": f"detr_matscholar_holdout_{HOLDOUT}.pt",
            "train_pt": f"dataset_matscholar_train_no_holdout_{HOLDOUT}.pt",
            "train_csv": f"env_targets_matscholar_train_no_holdout_{HOLDOUT}.csv",
            "metadata_csv": "polymer_matscholar_env_vectors.csv",
        },
    },
    "SOAP": {
        "full": {
            "checkpoint": "detr_soap.pt",
            "train_pt": "dataset_soap.pt",
            "train_csv": "env_targets_soap.csv",
            "metadata_csv": "polymer_skipatom_env_vectors.csv",
        },
        "holdout": {
            "checkpoint": f"detr_soap_holdout_{HOLDOUT}.pt",
            "train_pt": f"dataset_soap_train_no_holdout_{HOLDOUT}.pt",
            "train_csv": f"env_targets_soap_train_no_holdout_{HOLDOUT}.csv",
            "metadata_csv": "polymer_skipatom_env_vectors.csv",
        },
    },
}


# ------------------------------------------------------------
# CHEMISTRY HELPERS
# ------------------------------------------------------------
def canonicalise_polymer_name(x):
    if pd.isna(x):
        return None

    s = str(x).strip().lower()
    s = re.sub(r"\([^)]*\)", "", s)

    if s.startswith("poly(") and s.endswith(")"):
        s = s[5:-1].strip()

    s = re.sub(r"^poly[-\s]+", "", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


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


def has_neighbour(atom, atomic_num):
    return any(n.GetAtomicNum() == atomic_num for n in atom.GetNeighbors())


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


def classify_environment(smiles, atom_index):
    mol = mol_from_smiles(smiles)
    atom = get_atom(mol, atom_index)

    if atom is None:
        return "unknown_env"

    z = atom.GetAtomicNum()

    if z == 6:
        if atom.GetIsAromatic():
            return "aromatic C"
        if has_neighbour(atom, 35):
            return "C-Br"
        if has_neighbour(atom, 17):
            return "C-Cl"
        if has_neighbour(atom, 9):
            return "C-F"

        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() == 2.0:
                return "carbonyl C"

        if has_neighbour(atom, 8):
            return "C-O"
        if has_neighbour(atom, 7):
            return "C-N"

        return "aliphatic C"

    if z == 8:
        if is_carbonyl_oxygen(smiles, atom_index):
            return "carbonyl O"
        return "other O"

    if z == 7:
        return "N"
    if z == 35:
        return "Br"
    if z == 17:
        return "Cl"
    if z == 9:
        return "F"
    if z == 16:
        return "S"
    if z == 15:
        return "P"

    return atom.GetSymbol()


def coarse_env(label):
    label = str(label)

    if "C-Br" in label:
        return "C-Br"
    if "C-Cl" in label:
        return "C-Cl"
    if "C-F" in label:
        return "C-F"
    if "carbonyl O" in label:
        return "carbonyl O"
    if "carbonyl C" in label:
        return "carbonyl C"
    if "C-O" in label:
        return "C-O"
    if "C-N" in label:
        return "C-N"
    if "aromatic C" in label:
        return "aromatic C"
    if "aliphatic C" in label:
        return "aliphatic C"
    if "other O" in label:
        return "other O"
    if label.startswith("N"):
        return "N"
    if label.startswith("S"):
        return "S"
    if label.startswith("P"):
        return "P"
    if label.startswith("Cl"):
        return "Cl"
    if label.startswith("Br"):
        return "Br"
    if label.startswith("F"):
        return "F"

    return "unknown/other"


# ------------------------------------------------------------
# STRUCTURE-DERIVED GROUND TRUTH
# ------------------------------------------------------------
def derive_ground_truth_envs_from_smiles(smiles):
    mol = mol_from_smiles(smiles)

    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    env_rows = []
    counts = Counter()

    for atom in mol.GetAtoms():
        env = classify_environment(smiles, atom.GetIdx())
        counts[env] += 1
        env_rows.append({
            "atom_index": atom.GetIdx(),
            "atom_symbol": atom.GetSymbol(),
            "environment_label": env,
        })

    atom_df = pd.DataFrame(env_rows)

    summary_df = (
        pd.DataFrame([
            {"environment_label": label, "true_count": count}
            for label, count in counts.items()
        ])
        .sort_values(["true_count", "environment_label"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return atom_df, summary_df


def build_ground_truth(polymer_smiles=POLYMER_SMILES):
    ground_truth = {}

    for name, smiles in polymer_smiles.items():
        atom_df, summary_df = derive_ground_truth_envs_from_smiles(smiles)
        ground_truth[name] = {
            "atom_df": atom_df,
            "summary_df": summary_df,
        }

    return ground_truth


# ------------------------------------------------------------
# VGD PARSING AND PREPROCESSING
# ------------------------------------------------------------
def load_vgd_spectrum(vgd_path, use_corrected=False, spectrum_index=0):
    vgd_path = Path(vgd_path)
    data = read_vgd(str(vgd_path))

    if hasattr(data, "spectra") and data.spectra is not None and len(data.spectra) > 0:
        spec = data.spectra[spectrum_index]
        energy = np.asarray(spec.binding_energy, dtype=float)
        intensity = np.asarray(
            spec.corrected_intensity if use_corrected else spec.intensity,
            dtype=float,
        )
        meta = {
            "core_level": getattr(spec, "core_level", None),
            "be_start": getattr(spec, "be_start", None),
            "be_end": getattr(spec, "be_end", None),
            "be_step": getattr(spec, "be_step", None),
            "pass_energy": getattr(spec, "pass_energy", None),
            "source_energy": getattr(spec, "source_energy", None),
            "spectrum_index": getattr(spec, "spectrum_index", spectrum_index),
        }
    else:
        energy = np.asarray(data.binding_energy, dtype=float)
        intensity = np.asarray(
            data.corrected_intensity if use_corrected else data.intensity,
            dtype=float,
        )
        meta = {
            "core_level": getattr(data, "core_level", None),
            "pass_energy": getattr(getattr(data, "acquisition", None), "pass_energy", None),
            "source_energy": getattr(getattr(data, "acquisition", None), "source_energy", None),
            "spectrum_index": spectrum_index,
        }

    order = np.argsort(energy)
    energy = energy[order]
    intensity = intensity[order]

    return energy, intensity, meta


def preprocess_xps_spectrum(energy, intensity, target_len=TARGET_LEN, normalise=True):
    energy = np.asarray(energy, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    mask = np.isfinite(energy) & np.isfinite(intensity)
    energy = energy[mask]
    intensity = intensity[mask]

    if len(energy) < 10:
        raise ValueError("Too few valid points in parsed spectrum.")

    unique_energy, unique_idx = np.unique(energy, return_index=True)
    energy = unique_energy
    intensity = intensity[unique_idx]

    e_new = np.linspace(energy.min(), energy.max(), target_len)
    y_new = np.interp(e_new, energy, intensity)

    if normalise:
        y_new = y_new - np.min(y_new)
        ymax = np.max(y_new)
        if ymax > 0:
            y_new = y_new / ymax

    return e_new, y_new


# ------------------------------------------------------------
# MODEL AND TRAINING-METADATA HELPERS
# ------------------------------------------------------------
def load_model_for_env_inspection(checkpoint_path, device=DEVICE):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = SpectrumDETR(n_queries=24)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def load_aligned_train_envs(train_pt_path, train_csv_path, metadata_csv):
    data = torch.load(train_pt_path, map_location="cpu")
    env_df = pd.read_csv(train_csv_path).reset_index(drop=True)
    meta_df = pd.read_csv(metadata_csv).reset_index(drop=True)

    env_rows = []

    if isinstance(data, list):
        for i, example in enumerate(data):
            if "env_embeddings" not in example:
                raise KeyError(f"Example {i} missing 'env_embeddings'")

            env = example["env_embeddings"]

            if torch.is_tensor(env):
                env = env.detach().cpu().numpy()
            else:
                env = np.asarray(env, dtype=np.float32)

            if env.ndim == 1:
                env = env[None, :]

            env_rows.append(env)
    else:
        raise ValueError("Expected training PT to be a list of dict examples.")

    x_train = np.vstack(env_rows).astype(np.float32)

    if len(x_train) != len(env_df):
        raise ValueError(
            f"Mismatch: embeddings={len(x_train)} rows, env_df={len(env_df)} rows "
            f"for {train_pt_path} and {train_csv_path}"
        )

    train_meta = env_df.copy()

    if "name" in meta_df.columns:
        meta_df["meta_norm_key"] = meta_df["name"].astype(str).map(canonicalise_polymer_name)
    else:
        meta_df["meta_norm_key"] = None

    if "norm_key" in train_meta.columns:
        train_meta["norm_key_join"] = train_meta["norm_key"].astype(str).map(canonicalise_polymer_name)
    else:
        train_meta["norm_key_join"] = None

    if "atom_index" in meta_df.columns:
        meta_df["atom_index_join"] = pd.to_numeric(meta_df["atom_index"], errors="coerce")
    else:
        meta_df["atom_index_join"] = np.nan

    if "env_index_within_polymer" in train_meta.columns:
        train_meta["env_index_join"] = pd.to_numeric(train_meta["env_index_within_polymer"], errors="coerce")
    else:
        train_meta["env_index_join"] = np.nan

    keep_cols = [
        col for col in
        ["meta_norm_key", "atom_index_join", "smiles", "atom_symbol", "atom_index", "name"]
        if col in meta_df.columns
    ]

    meta_small = meta_df[keep_cols].copy()

    train_meta = train_meta.merge(
        meta_small,
        left_on=["norm_key_join", "env_index_join"],
        right_on=["meta_norm_key", "atom_index_join"],
        how="left",
    )

    return x_train, train_meta


def make_env_label(row):
    smiles = row["smiles"] if "smiles" in row.index and pd.notna(row["smiles"]) else None

    atom_index = None
    for col in ["atom_index", "env_index_within_polymer", "atom_idx"]:
        if col in row.index and pd.notna(row[col]):
            atom_index = row[col]
            break

    env = classify_environment(smiles, atom_index)

    extras = []

    if "atom_symbol" in row.index and pd.notna(row["atom_symbol"]):
        extras.append(f"atom={row['atom_symbol']}")

    if atom_index is not None and pd.notna(atom_index):
        extras.append(f"idx={int(atom_index)}")

    return f"{env} | " + " | ".join(extras) if extras else env


def cosine_similarity_manual(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)

    return a @ b.T


# ------------------------------------------------------------
# PREDICTION AND COMPARISON
# ------------------------------------------------------------
def summarise_predicted_coarse_envs_from_vgd(
    polymer_name,
    checkpoint_path,
    train_pt_path,
    train_csv_path,
    metadata_csv,
    use_corrected=False,
    conf_thresh=CONF_THRESH,
    show_plots=True,
    experimental_dir=EXPERIMENTAL_DIR,
    device=DEVICE,
):
    vgd_path = Path(experimental_dir) / polymer_name / "C1s Scan.VGD"

    if not vgd_path.exists():
        raise FileNotFoundError(vgd_path)

    energy, intensity, meta = load_vgd_spectrum(vgd_path, use_corrected=use_corrected)
    e_plot, y_plot = preprocess_xps_spectrum(
        energy,
        intensity,
        target_len=TARGET_LEN,
        normalise=True,
    )

    model = load_model_for_env_inspection(checkpoint_path, device=device)

    x_tensor = (
        torch.tensor(y_plot, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        env_preds, conf_logits = model(x_tensor)

    env_preds = env_preds[0].detach().cpu().numpy()
    conf = torch.sigmoid(conf_logits[0]).detach().cpu().numpy()

    keep = conf > conf_thresh
    kept_envs = env_preds[keep]
    kept_conf = conf[keep]
    kept_query_ids = np.where(keep)[0]

    if show_plots:
        print("VGD meta:", meta)
        print("Conf stats:", {
            "min": float(conf.min()),
            "max": float(conf.max()),
            "mean": float(conf.mean()),
            "n_kept": int(keep.sum()),
        })

        plt.figure(figsize=(8, 4))
        plt.plot(e_plot, y_plot, linewidth=1.2)
        plt.gca().invert_xaxis()
        plt.xlabel("Binding Energy (eV)")
        plt.ylabel("Normalised intensity")
        plt.title(f"{polymer_name} | parsed VGD spectrum")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 3))
        plt.bar(np.arange(len(conf)), conf)
        plt.axhline(conf_thresh, linestyle="--")
        plt.title(f"{polymer_name} | query confidences")
        plt.xlabel("Query index")
        plt.ylabel("Confidence")
        plt.tight_layout()
        plt.show()

    if len(kept_envs) == 0:
        return (
            pd.DataFrame(columns=[
                "environment_label",
                "predicted_count",
                "mean_query_confidence",
                "mean_cosine_similarity",
            ]),
            pd.DataFrame(),
        )

    x_train, meta_train = load_aligned_train_envs(
        train_pt_path,
        train_csv_path,
        metadata_csv,
    )

    sims = cosine_similarity_manual(kept_envs, x_train)

    rows = []

    for qid, qconf, sim_row in zip(kept_query_ids, kept_conf, sims):
        best_idx = np.argsort(sim_row)[::-1][0]
        row = meta_train.iloc[best_idx]
        fine_label = make_env_label(row)

        rows.append({
            "query_id": int(qid),
            "query_confidence": float(qconf),
            "best_match_label": fine_label,
            "environment_label": coarse_env(fine_label),
            "best_match_cosine_sim": float(sim_row[best_idx]),
        })

    pred_df = pd.DataFrame(rows)

    pred_summary_df = (
        pred_df.groupby("environment_label", as_index=False)
        .agg(
            predicted_count=("query_id", "count"),
            mean_query_confidence=("query_confidence", "mean"),
            mean_cosine_similarity=("best_match_cosine_sim", "mean"),
        )
        .sort_values(["predicted_count", "mean_query_confidence"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return pred_summary_df, pred_df


def compare_predicted_to_ground_truth(pred_summary_df, true_summary_df):
    comp = true_summary_df.merge(
        pred_summary_df,
        on="environment_label",
        how="outer",
    ).fillna({
        "true_count": 0,
        "predicted_count": 0,
        "mean_query_confidence": 0.0,
        "mean_cosine_similarity": 0.0,
    })

    comp["true_count"] = comp["true_count"].astype(int)
    comp["predicted_count"] = comp["predicted_count"].astype(int)

    comp = comp.sort_values(
        ["true_count", "predicted_count", "environment_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return comp


def run_one_case(
    polymer_name,
    representation,
    regime,
    use_corrected=False,
    conf_thresh=CONF_THRESH,
    show_plots=True,
    ground_truth=None,
):
    if ground_truth is None:
        ground_truth = build_ground_truth()

    cfg = REP_CONFIGS[representation][regime]

    print("\n" + "=" * 90)
    print(f"{polymer_name} | {representation} | {regime}")
    print("=" * 90)

    true_summary_df = ground_truth[polymer_name]["summary_df"]

    pred_summary_df, pred_df = summarise_predicted_coarse_envs_from_vgd(
        polymer_name=polymer_name,
        checkpoint_path=cfg["checkpoint"],
        train_pt_path=cfg["train_pt"],
        train_csv_path=cfg["train_csv"],
        metadata_csv=cfg["metadata_csv"],
        use_corrected=use_corrected,
        conf_thresh=conf_thresh,
        show_plots=show_plots,
    )

    comparison = compare_predicted_to_ground_truth(
        pred_summary_df=pred_summary_df,
        true_summary_df=true_summary_df,
    )

    return {
        "polymer": polymer_name,
        "representation": representation,
        "regime": regime,
        "true_summary_df": true_summary_df,
        "pred_summary_df": pred_summary_df,
        "pred_df": pred_df,
        "comparison": comparison,
    }


def run_all_cases(
    polymers=None,
    representations=None,
    regimes=None,
    use_corrected=False,
    conf_thresh=CONF_THRESH,
    show_plots=False,
):
    ground_truth = build_ground_truth()

    if polymers is None:
        polymers = list(POLYMER_SMILES.keys())

    if representations is None:
        representations = list(REP_CONFIGS.keys())

    if regimes is None:
        regimes = ["full", "holdout"]

    outputs = []
    summary_rows = []

    for polymer_name in polymers:
        for representation in representations:
            for regime in regimes:
                output = run_one_case(
                    polymer_name=polymer_name,
                    representation=representation,
                    regime=regime,
                    use_corrected=use_corrected,
                    conf_thresh=conf_thresh,
                    show_plots=show_plots,
                    ground_truth=ground_truth,
                )

                outputs.append(output)

                comp = output["comparison"].copy()

                true_total = int(comp["true_count"].sum())
                pred_total = int(comp["predicted_count"].sum())
                matched_mass = int(np.minimum(
                    comp["true_count"].values,
                    comp["predicted_count"].values,
                ).sum())

                true_classes = set(comp.loc[comp["true_count"] > 0, "environment_label"])
                pred_classes = set(comp.loc[comp["predicted_count"] > 0, "environment_label"])

                summary_rows.append({
                    "polymer": polymer_name,
                    "representation": representation,
                    "regime": regime,
                    "true_total_envs": true_total,
                    "predicted_total_envs": pred_total,
                    "matched_count_mass": matched_mass,
                    "matched_fraction_of_true": matched_mass / true_total if true_total > 0 else np.nan,
                    "n_true_classes": len(true_classes),
                    "n_pred_classes": len(pred_classes),
                    "n_overlap_classes": len(true_classes & pred_classes),
                    "missed_classes": ", ".join(sorted(true_classes - pred_classes)),
                    "hallucinated_classes": ", ".join(sorted(pred_classes - true_classes)),
                })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["polymer", "representation", "regime"],
    ).reset_index(drop=True)

    return outputs, summary_df


if __name__ == "__main__":
    outputs, summary_df = run_all_cases(show_plots=False)
    print(summary_df)
