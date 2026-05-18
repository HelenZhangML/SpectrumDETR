"""
Build synthetic/augmented dataset variants from real polymer XPS datasets.

Inputs:
    dataset_{rep}.pt
    dataset_meta_{rep}.json
    dataset_{rep}_{split_mode}_{holdout}.pt
    dataset_meta_{rep}_{split_mode}_{holdout}.json

Outputs:
    dataset_{rep}_synth.pt
    dataset_meta_{rep}_synth.json
    dataset_{rep}_synth_{split_mode}_{holdout}.pt
    dataset_meta_{rep}_synth_{split_mode}_{holdout}.json
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ============================================================
# DEFAULT CONFIG
# ============================================================

DATASET_BUILDS = [
    ("all", None),

    ("train_no_holdout", "CBr"),
    ("test_only_holdout", "CBr"),

    ("train_no_holdout", "carbonyl_O"),
    ("test_only_holdout", "carbonyl_O"),

    ("train_no_holdout", "aromatic_C"),
    ("test_only_holdout", "aromatic_C"),
]

DEFAULT_AUG_KWARGS = dict(
    shift_std=0.05,
    blur_sigma_range=(0.0, 0.5),
    noise_std_range=(0.0, 0.015),
    scale_range=(0.95, 1.05),
)


# ============================================================
# BASIC HELPERS
# ============================================================

def get_key(ex: Dict) -> str:
    key = ex.get("key", "")
    if not isinstance(key, str):
        key = str(key)
    return key


def real_dataset_paths(
    rep: str,
    split_mode: str,
    holdout_name: Optional[str],
) -> Tuple[Path, Path]:
    if split_mode == "all":
        return (
            Path(f"dataset_{rep}.pt"),
            Path(f"dataset_meta_{rep}.json"),
        )

    suffix = f"_{split_mode}_{holdout_name}"

    return (
        Path(f"dataset_{rep}{suffix}.pt"),
        Path(f"dataset_meta_{rep}{suffix}.json"),
    )


def synthetic_dataset_paths(
    rep: str,
    split_mode: str,
    holdout_name: Optional[str],
) -> Tuple[Path, Path]:
    if split_mode == "all":
        return (
            Path(f"dataset_{rep}_synth.pt"),
            Path(f"dataset_meta_{rep}_synth.json"),
        )

    suffix = f"_{split_mode}_{holdout_name}"

    return (
        Path(f"dataset_{rep}_synth{suffix}.pt"),
        Path(f"dataset_meta_{rep}_synth{suffix}.json"),
    )


# ============================================================
# AUGMENTATION
# ============================================================

def normalise(y: np.ndarray, mode: str = "max") -> np.ndarray:
    y = y.astype(np.float32)

    if mode == "max":
        m = float(np.max(np.abs(y)) + 1e-12)
        return (y / m).astype(np.float32)

    if mode == "zscore":
        return ((y - y.mean()) / (y.std() + 1e-12)).astype(np.float32)

    return y.astype(np.float32)


def gaussian_blur_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 1e-6:
        return y.astype(np.float32)

    radius = int(max(3, np.ceil(4 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)

    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum() + 1e-12

    return np.convolve(y, kernel, mode="same").astype(np.float32)


def augment_spectrum(
    x_grid: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    shift_std: float = 0.05,
    blur_sigma_range=(0.0, 0.5),
    noise_std_range=(0.0, 0.015),
    scale_range=(0.95, 1.05),
) -> np.ndarray:
    y = y.astype(np.float32)

    # small binding-energy calibration shift
    if shift_std > 0:
        shift = float(rng.normal(0.0, shift_std))
        y = np.interp(
            x_grid,
            x_grid + shift,
            y,
            left=y[0],
            right=y[-1],
        ).astype(np.float32)

    # light broadening
    lo, hi = blur_sigma_range
    if hi > 0:
        sigma = float(rng.uniform(lo, hi))
        y = gaussian_blur_1d(y, sigma)

    # intensity scaling
    scale = float(rng.uniform(*scale_range))
    y = (y * scale).astype(np.float32)

    # additive noise
    noise_std = float(rng.uniform(*noise_std_range))
    if noise_std > 0:
        y = (
            y
            + rng.normal(0.0, noise_std, size=y.shape)
        ).astype(np.float32)

    return y.astype(np.float32)


def safe_augment(
    x_grid: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    normalise_mode: str = "max",
    **aug_kwargs,
) -> np.ndarray:
    y_aug = augment_spectrum(
        x_grid,
        y.astype(np.float32),
        rng,
        **aug_kwargs,
    )

    return normalise(y_aug, normalise_mode)


# ============================================================
# SPLITTING
# ============================================================

def polymer_level_split(
    ds_real: List[Dict],
    rng: np.random.Generator,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Tuple[List[int], List[int], List[int], Dict[str, List[str]]]:

    poly_to_indices: Dict[str, List[int]] = {}

    for i, ex in enumerate(ds_real):
        key = get_key(ex)

        if not key:
            raise ValueError(
                "Every real example must have a non-empty 'key' "
                "for polymer-level splitting."
            )

        poly_to_indices.setdefault(key, []).append(i)

    poly_keys = sorted(poly_to_indices.keys())
    n_poly = len(poly_keys)

    if n_poly < 3:
        raise ValueError(f"Too few polymers for splitting: n_poly={n_poly}")

    perm = rng.permutation(n_poly)

    n_train = int(round(train_frac * n_poly))
    n_val = int(round(val_frac * n_poly))
    n_test = n_poly - n_train - n_val

    if n_test < 0:
        n_test = 0
        n_val = n_poly - n_train

    train_polys = [poly_keys[i] for i in perm[:n_train]]
    val_polys = [poly_keys[i] for i in perm[n_train:n_train + n_val]]
    test_polys = [poly_keys[i] for i in perm[n_train + n_val:]]

    def indices_for_polys(polys: List[str]) -> List[int]:
        indices = []
        for key in polys:
            indices.extend(poly_to_indices[key])
        return indices

    return (
        indices_for_polys(train_polys),
        indices_for_polys(val_polys),
        indices_for_polys(test_polys),
        {
            "train": train_polys,
            "val": val_polys,
            "test": test_polys,
        },
    )


# ============================================================
# BUILD ONE SYNTHETIC DATASET
# ============================================================

def build_synthetic_dataset(
    rep: str,
    split_mode: str,
    holdout_name: Optional[str],
    *,
    seed: int = 123,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    aug_per_real_train: int = 30,
    aug_per_real_val: int = 0,
    aug_per_real_test: int = 0,
    aug_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> None:

    if aug_kwargs is None:
        aug_kwargs = DEFAULT_AUG_KWARGS

    real_pt, real_meta = real_dataset_paths(
        rep,
        split_mode,
        holdout_name,
    )

    out_pt, out_meta = synthetic_dataset_paths(
        rep,
        split_mode,
        holdout_name,
    )

    if verbose:
        print("\n" + "=" * 70)
        print(
            f"Building synthetic dataset: "
            f"REP={rep}, SPLIT_MODE={split_mode}, HOLDOUT_NAME={holdout_name}"
        )
        print("=" * 70)
        print("REAL_PT:", real_pt)
        print("REAL_META:", real_meta)

    if not real_pt.exists():
        raise FileNotFoundError(f"Missing real dataset: {real_pt}")

    if not real_meta.exists():
        raise FileNotFoundError(f"Missing real metadata: {real_meta}")

    ds_real = torch.load(real_pt, map_location="cpu")

    with open(real_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if len(ds_real) == 0:
        raise ValueError(f"Empty real dataset: {real_pt}")

    x_grid = np.linspace(
        meta["x_min"],
        meta["x_max"],
        meta["n_points"],
    ).astype(np.float32)

    normalise_mode = meta.get("normalise_mode", "max")

    if verbose:
        print("Real examples:", len(ds_real))
        print("Spectrum shape:", tuple(ds_real[0]["spectrum"].shape))
        print("Env shape:", tuple(ds_real[0]["env_embeddings"].shape))
        print(
            "x_grid:",
            (float(x_grid.min()), float(x_grid.max())),
            "N =",
            len(x_grid),
        )

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------
    if split_mode == "test_only_holdout":
        idx_train = []
        idx_val = []
        idx_test = list(range(len(ds_real)))

        split_polys = {
            "train": [],
            "val": [],
            "test": sorted({get_key(ex) for ex in ds_real}),
        }

        if verbose:
            print("OOD dataset: using all held-out polymers as evaluation set.")
            print("Held-out polymers:", len(split_polys["test"]))
            print("Held-out real examples:", len(idx_test))

    else:
        idx_train, idx_val, idx_test, split_polys = polymer_level_split(
            ds_real,
            rng,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )

        if verbose:
            print(
                "Polymer split:",
                {
                    "train": len(split_polys["train"]),
                    "val": len(split_polys["val"]),
                    "test": len(split_polys["test"]),
                },
            )
            print(
                "Real examples:",
                {
                    "train": len(idx_train),
                    "val": len(idx_val),
                    "test": len(idx_test),
                },
            )

    # --------------------------------------------------------
    # Build synthetic dataset
    # --------------------------------------------------------
    ds_synth = []

    def add_with_aug(
        indices: List[int],
        split_name: str,
        aug_per_real: int,
        rng_local: np.random.Generator,
    ) -> None:
        for i in indices:
            ex = ds_real[i]

            y = (
                ex["spectrum"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            env = ex["env_embeddings"]

            # original example
            ds_synth.append({
                "key": ex.get("key", ""),
                "spectrum": ex["spectrum"].clone(),
                "env_embeddings": env.clone(),
                "kind": "real",
                "split": split_name,
                "source_idx": int(i),
                "aug_id": 0,
                "is_augmented": False,
            })

            # augmented copies
            for aug_id in range(1, int(aug_per_real) + 1):
                y_aug = safe_augment(
                    x_grid,
                    y,
                    rng_local,
                    normalise_mode=normalise_mode,
                    **aug_kwargs,
                )

                ds_synth.append({
                    "key": ex.get("key", ""),
                    "spectrum": torch.from_numpy(y_aug),
                    "env_embeddings": env.clone(),
                    "kind": "aug",
                    "split": split_name,
                    "source_idx": int(i),
                    "aug_id": int(aug_id),
                    "is_augmented": True,
                })

    rng_train = np.random.default_rng(seed + 1)
    rng_val = np.random.default_rng(seed + 2)
    rng_test = np.random.default_rng(seed + 3)

    if split_mode == "test_only_holdout":
        add_with_aug(idx_test, "ood", 0, rng_test)
    else:
        add_with_aug(idx_train, "train", aug_per_real_train, rng_train)
        add_with_aug(idx_val, "val", aug_per_real_val, rng_val)
        add_with_aug(idx_test, "test", aug_per_real_test, rng_test)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    torch.save(ds_synth, out_pt)

    output_meta = {
        "rep": rep,
        "source_real_dataset": str(real_pt),
        "split_mode": split_mode,
        "holdout_name": holdout_name,
        "n_real_examples": len(ds_real),
        "n_synth_examples": len(ds_synth),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "split_polymers": split_polys,
        "aug_per_real_train": aug_per_real_train,
        "aug_per_real_val": aug_per_real_val,
        "aug_per_real_test": aug_per_real_test,
        "augmentation": aug_kwargs,
        "x_min": float(meta["x_min"]),
        "x_max": float(meta["x_max"]),
        "n_points": int(meta["n_points"]),
        "normalise_mode": normalise_mode,
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2)

    if verbose:
        print("Saved synthetic dataset:", out_pt)
        print("Saved synthetic metadata:", out_meta)


# ============================================================
# BUILD ALL VARIANTS
# ============================================================

def build_all_synthetic_datasets(
    rep: str,
    *,
    dataset_builds=DATASET_BUILDS,
    seed: int = 123,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    aug_per_real_train: int = 30,
    aug_per_real_val: int = 0,
    aug_per_real_test: int = 0,
    aug_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> None:

    for split_mode, holdout_name in dataset_builds:
        build_synthetic_dataset(
            rep=rep,
            split_mode=split_mode,
            holdout_name=holdout_name,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            aug_per_real_train=aug_per_real_train,
            aug_per_real_val=aug_per_real_val,
            aug_per_real_test=aug_per_real_test,
            aug_kwargs=aug_kwargs,
            verbose=verbose,
        )


# ============================================================
# VALIDATION / SANITY CHECKS
# ============================================================

def validate_synthetic_dataset(
    rep: str,
    split_mode: str,
    holdout_name: Optional[str],
    *,
    verbose: bool = True,
) -> Dict:

    out_pt, out_meta = synthetic_dataset_paths(
        rep,
        split_mode,
        holdout_name,
    )

    if verbose:
        print("\n" + "-" * 70)
        print("Checking:", out_pt)

    if not out_pt.exists():
        raise FileNotFoundError(f"Missing dataset file: {out_pt}")

    if not out_meta.exists():
        raise FileNotFoundError(f"Missing metadata file: {out_meta}")

    ds = torch.load(out_pt, map_location="cpu")

    with open(out_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if len(ds) == 0:
        raise ValueError(f"Empty synthetic dataset: {out_pt}")

    x_grid = np.linspace(
        meta["x_min"],
        meta["x_max"],
        meta["n_points"],
    ).astype(np.float32)

    split_counts = {}
    kind_counts = {}

    bad_shapes = 0
    nan_spec = 0
    nan_tgt = 0

    for ex in ds:
        split = ex.get("split", "none")
        kind = ex.get("kind", "none")

        split_counts[split] = split_counts.get(split, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

        spectrum = ex["spectrum"]
        target = ex["env_embeddings"]

        if spectrum.ndim != 1 or spectrum.shape[0] != len(x_grid):
            bad_shapes += 1

        if torch.isnan(spectrum).any():
            nan_spec += 1

        if torch.isnan(target).any():
            nan_tgt += 1

    summary = {
        "dataset": str(out_pt),
        "metadata": str(out_meta),
        "n_examples": len(ds),
        "keys": list(ds[0].keys()),
        "split_counts": split_counts,
        "kind_counts": kind_counts,
        "bad_spectrum_shapes": bad_shapes,
        "nan_spectra": nan_spec,
        "nan_targets": nan_tgt,
    }

    if verbose:
        print("Total:", summary["n_examples"])
        print("Keys:", summary["keys"])
        print("Splits:", split_counts)
        print("Kinds:", kind_counts)
        print("Bad spectrum shapes:", bad_shapes)
        print("NaN spectra:", nan_spec)
        print("NaN targets:", nan_tgt)

    return summary


def validate_all_synthetic_datasets(
    rep: str,
    *,
    dataset_builds=DATASET_BUILDS,
    verbose: bool = True,
) -> List[Dict]:

    summaries = []

    for split_mode, holdout_name in dataset_builds:
        summaries.append(
            validate_synthetic_dataset(
                rep=rep,
                split_mode=split_mode,
                holdout_name=holdout_name,
                verbose=verbose,
            )
        )

    return summaries


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    build_all_synthetic_datasets(rep="skipatom")
    validate_all_synthetic_datasets(rep="skipatom")
