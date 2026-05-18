import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine


REP_CSVS = {
    "SkipAtom": ("polymer_skipatom_env_vectors.csv", "env_"),
    "MatScholar": ("polymer_matscholar_env_vectors.csv", "matscholar_env_"),
    "SOAP": ("soap_env_vectors.csv", "soap_env_"),
}

PCA_DIM = 50
PERPLEXITY = 40
RANDOM_STATE = 0


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


def bonded_to(atom, atomic_num):
    if atom is None:
        return False
    return any(n.GetAtomicNum() == atomic_num for n in atom.GetNeighbors())


def is_carbonyl_carbon(atom):
    if atom is None or atom.GetAtomicNum() != 6:
        return False
    for bond in atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 8:
                return True
    return False


def is_carbonyl_oxygen(atom):
    if atom is None or atom.GetAtomicNum() != 8:
        return False
    for bond in atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 6:
                return True
    return False


def mask_C_hal(df, atomic_num):
    mask = np.zeros(len(df), dtype=bool)

    for i, (smi, idx) in enumerate(zip(df["smiles"], df["atom_index"])):
        mol = mol_from_smiles(smi)
        atom = get_atom(mol, idx)

        if atom is not None and atom.GetAtomicNum() == 6 and bonded_to(atom, atomic_num):
            mask[i] = True

    return mask


def mask_C_f(df):
    return mask_C_hal(df, 9)


def mask_C_cl(df):
    return mask_C_hal(df, 17)


def mask_C_br(df):
    return mask_C_hal(df, 35)


def mask_C_aromatic(df):
    mask = np.zeros(len(df), dtype=bool)

    for i, (smi, idx) in enumerate(zip(df["smiles"], df["atom_index"])):
        mol = mol_from_smiles(smi)
        atom = get_atom(mol, idx)

        if atom is not None and atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
            mask[i] = True

    return mask


def mask_C_sp3(df):
    mask = np.zeros(len(df), dtype=bool)

    for i, (smi, idx) in enumerate(zip(df["smiles"], df["atom_index"])):
        mol = mol_from_smiles(smi)
        atom = get_atom(mol, idx)

        if (
            atom is not None
            and atom.GetAtomicNum() == 6
            and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3
        ):
            mask[i] = True

    return mask


def mask_C_carbonyl(df):
    mask = np.zeros(len(df), dtype=bool)

    for i, (smi, idx) in enumerate(zip(df["smiles"], df["atom_index"])):
        mol = mol_from_smiles(smi)
        atom = get_atom(mol, idx)

        if is_carbonyl_carbon(atom):
            mask[i] = True

    return mask


def mask_O_carbonyl(df):
    mask = np.zeros(len(df), dtype=bool)

    for i, (smi, idx) in enumerate(zip(df["smiles"], df["atom_index"])):
        mol = mol_from_smiles(smi)
        atom = get_atom(mol, idx)

        if is_carbonyl_oxygen(atom):
            mask[i] = True

    return mask


def mask_O_ether(df):
    mask = np.zeros(len(df), dtype=bool)

    for i, (smi, idx) in enumerate(zip(df["smiles"], df["atom_index"])):
        mol = mol_from_smiles(smi)
        atom = get_atom(mol, idx)

        if atom is None or atom.GetAtomicNum() != 8:
            continue

        c_neighbours = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        if c_neighbours >= 2 and not is_carbonyl_oxygen(atom):
            mask[i] = True

    return mask


def mask_N_amide(df):
    mask = np.zeros(len(df), dtype=bool)

    for i, (smi, idx) in enumerate(zip(df["smiles"], df["atom_index"])):
        mol = mol_from_smiles(smi)
        atom = get_atom(mol, idx)

        if atom is None or atom.GetAtomicNum() != 7:
            continue

        if any(n.GetAtomicNum() == 6 and is_carbonyl_carbon(n) for n in atom.GetNeighbors()):
            mask[i] = True

    return mask


def mask_N_aromatic(df):
    mask = np.zeros(len(df), dtype=bool)

    for i, (smi, idx) in enumerate(zip(df["smiles"], df["atom_index"])):
        mol = mol_from_smiles(smi)
        atom = get_atom(mol, idx)

        if atom is not None and atom.GetAtomicNum() == 7 and atom.GetIsAromatic():
            mask[i] = True

    return mask


MASKS = {
    "C–F": mask_C_f,
    "C–Cl": mask_C_cl,
    "C–Br": mask_C_br,
    "C(aromatic)": mask_C_aromatic,
    "C(sp3)": mask_C_sp3,
    "C(carbonyl)": mask_C_carbonyl,
    "O(carbonyl)": mask_O_carbonyl,
    "O(ether)": mask_O_ether,
    "N(amide)": mask_N_amide,
    "N(aromatic)": mask_N_aromatic,
}


def compute_tsne(X):
    X_scaled = StandardScaler().fit_transform(X)

    if X_scaled.shape[1] > PCA_DIM:
        X_reduced = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE).fit_transform(X_scaled)
    else:
        X_reduced = X_scaled

    Z = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
    ).fit_transform(X_reduced)

    return X_scaled, Z


def safe_filename(text):
    text = text.replace("–", "-")
    text = re.sub(r"[^\w\-.]+", "_", text)
    return text.strip("_")


def load_representation(rep):
    csv_path, prefix = REP_CSVS[rep]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith(prefix)]

    if not feature_cols:
        raise ValueError(f"No feature columns starting with '{prefix}' found in {csv_path}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    X_scaled, Z = compute_tsne(X)

    return df, X_scaled, Z


def plot_all_tsne_pairs(
    pairs=None,
    out_dir="figures_clean_pairs_multi",
):
    if pairs is None:
        pairs = [
            ("C–F", "C–Cl"),
            ("C–Cl", "C–Br"),
            ("O(ether)", "O(carbonyl)"),
            ("N(amide)", "N(aromatic)"),
            ("C(aromatic)", "C(sp3)"),
            ("C(carbonyl)", "O(carbonyl)"),
        ]

    os.makedirs(out_dir, exist_ok=True)

    rows = []

    for rep in REP_CSVS:
        try:
            df, X_scaled, Z = load_representation(rep)
        except FileNotFoundError:
            print(f"Missing input for {rep}; skipping.")
            continue

        print(f"\n=== {rep} ===")
        print("X:", X_scaled.shape)

        for env_a, env_b in pairs:
            if env_a not in MASKS or env_b not in MASKS:
                print(f"Unknown pair: {env_a}, {env_b}")
                continue

            mask_a = MASKS[env_a](df)
            mask_b = MASKS[env_b](df)

            n_a = int(mask_a.sum())
            n_b = int(mask_b.sum())

            pair_label = f"{env_a} vs {env_b}"

            if n_a == 0 or n_b == 0:
                rows.append({
                    "representation": rep,
                    "pair": pair_label,
                    "n_A": n_a,
                    "n_B": n_b,
                    "cosine_dist": np.nan,
                })
                print(f"{pair_label}: missing n=({n_a}, {n_b})")
                continue

            dist = float(cosine(X_scaled[mask_a].mean(axis=0), X_scaled[mask_b].mean(axis=0)))

            rows.append({
                "representation": rep,
                "pair": pair_label,
                "n_A": n_a,
                "n_B": n_b,
                "cosine_dist": dist,
            })

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.07, color="grey")
            ax.scatter(Z[mask_a, 0], Z[mask_a, 1], s=8, alpha=0.9, color="red", label=f"{env_a} (n={n_a})")
            ax.scatter(Z[mask_b, 0], Z[mask_b, 1], s=8, alpha=0.9, color="blue", label=f"{env_b} (n={n_b})")

            ax.set_title(f"{rep}: {pair_label}")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend(frameon=False)

            plt.tight_layout()

            out_path = os.path.join(
                out_dir,
                safe_filename(f"{rep.lower()}__{env_a}_vs_{env_b}.png"),
            )

            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.show()

            print("Saved:", out_path)

    result_df = pd.DataFrame(rows).sort_values(["pair", "representation"]).reset_index(drop=True)
    table_path = os.path.join(out_dir, "pairwise_env_similarity.csv")
    result_df.to_csv(table_path, index=False)

    print("Saved table:", table_path)
    return result_df


def plot_tsne_grid(
    pairs=None,
    out_dir="figures_tsne_grid",
    out_name="tsne_3x3_column_legends.png",
):
    if pairs is None:
        pairs = [
            ("C–Cl", "C–Br"),
            ("O(ether)", "O(carbonyl)"),
            ("C(aromatic)", "C(sp3)"),
        ]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.08, hspace=0.10)

    col_legend_handles = []
    col_legend_labels = []

    for row_idx, rep in enumerate(REP_CSVS):
        df, _, Z = load_representation(rep)

        for col_idx, (env_a, env_b) in enumerate(pairs):
            ax = axes[row_idx, col_idx]

            mask_a = MASKS[env_a](df)
            mask_b = MASKS[env_b](df)

            ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.05, color="grey")
            h1 = ax.scatter(Z[mask_a, 0], Z[mask_a, 1], s=10, color="red")
            h2 = ax.scatter(Z[mask_b, 0], Z[mask_b, 1], s=10, color="blue")

            if row_idx == 0:
                ax.set_title(f"{env_a} vs {env_b}", fontsize=18, pad=18)

            if col_idx == 0:
                ax.set_ylabel(rep, fontsize=18)

            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                col_legend_handles.append((h1, h2))
                col_legend_labels.append((env_a, env_b))

    for col_idx in range(len(pairs)):
        h1, h2 = col_legend_handles[col_idx]
        label_a, label_b = col_legend_labels[col_idx]

        axes[0, col_idx].legend(
            [h1, h2],
            [label_a, label_b],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.18),
            frameon=False,
            ncol=2,
            fontsize=15,
            handlelength=1.2,
            columnspacing=1.4,
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", out_path)
    return out_path


if __name__ == "__main__":
    plot_tsne_grid()
    plot_all_tsne_pairs()