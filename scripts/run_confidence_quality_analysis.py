import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from scripts.dataset import XPSSetDataset, collate_set
from scripts.detr_model import SpectrumDETR


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path("uncertainty_analysis_repro_synth")
OUT_DIR.mkdir(exist_ok=True)

REPS = ["skipatom", "matscholar", "soap"]
HOLDOUTS = ["CBr", "aromatic_C", "carbonyl_O"]

BATCH_SIZE = 8
CONF_THRESHOLD = 0.5
N_QUARTILES = 4


def first_existing(paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return str(path)
    return None


def load_model_from_ckpt(ckpt_path, device=DEVICE):
    model = SpectrumDETR(
        d_model=256,
        d_env=128,
        n_queries=24,
        n_heads=8,
        n_layers=3,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model")
            or ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt
        )
    else:
        state = ckpt

    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_prediction_distances(dataset, model, device=DEVICE):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_set,
        num_workers=0,
    )

    rows = []

    for batch_idx, batch in enumerate(loader):
        x = batch["spectra"].to(device)
        targets = [t.to(device) for t in batch["targets"]]

        env_preds, conf_logits = model(x)
        conf = torch.sigmoid(conf_logits)

        for sample_idx, (pred_env, pred_conf, target) in enumerate(zip(env_preds, conf, targets)):
            keep = pred_conf > CONF_THRESHOLD
            pred_env = pred_env[keep]
            pred_conf = pred_conf[keep]

            if len(pred_env) == 0 or len(target) == 0:
                continue

            pred_norm = pred_env / (pred_env.norm(dim=1, keepdim=True) + 1e-8)
            target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)

            dist = 1.0 - torch.matmul(pred_norm, target_norm.T)
            nearest_dist = dist.min(dim=1).values.detach().cpu().numpy()
            conf_np = pred_conf.detach().cpu().numpy()

            for pred_idx, (confidence, distance) in enumerate(zip(conf_np, nearest_dist)):
                rows.append({
                    "batch_idx": batch_idx,
                    "sample_idx": sample_idx,
                    "prediction_idx": pred_idx,
                    "confidence": float(confidence),
                    "nearest_cosine_distance": float(distance),
                    "n_targets": int(len(target)),
                    "n_kept_preds": int(len(pred_env)),
                })

    return pd.DataFrame(rows)


def summarise_best_worst(df, rep, setting, dataset_path, ckpt_path, split_path=None):
    df = df.copy()

    df["quality_bin"] = pd.qcut(
        df["nearest_cosine_distance"],
        q=N_QUARTILES,
        labels=["Q1_best", "Q2", "Q3", "Q4_worst"],
        duplicates="drop",
    )

    best = df[df["quality_bin"] == "Q1_best"]
    worst = df[df["quality_bin"] == "Q4_worst"]

    return {
        "rep": rep,
        "setting": setting,
        "n_predictions_kept": len(df),
        "n_best": len(best),
        "n_worst": len(worst),
        "mean_conf_best": best["confidence"].mean(),
        "mean_conf_worst": worst["confidence"].mean(),
        "median_conf_best": best["confidence"].median(),
        "median_conf_worst": worst["confidence"].median(),
        "delta_mean_conf_best_minus_worst": best["confidence"].mean() - worst["confidence"].mean(),
        "mean_dist_best": best["nearest_cosine_distance"].mean(),
        "mean_dist_worst": worst["nearest_cosine_distance"].mean(),
        "dataset_path": dataset_path,
        "checkpoint_path": ckpt_path,
        "split_path": split_path,
        "conf_threshold": CONF_THRESHOLD,
    }


def get_eval_specs():
    specs = []

    for rep in REPS:
        specs.append({
            "rep": rep,
            "setting": "baseline",
            "dataset_path": first_existing([
                f"dataset_{rep}_synth.pt",
                f"dataset_{rep}_synth_seed0.pt",
            ]),
            "ckpt_path": first_existing([
                f"detr_{rep}_baseline.pt",
                f"detr_{rep}_baseline_seed0.pt",
            ]),
            "split_path": first_existing([
                f"split_{rep}_baseline_synth.json",
                f"split_{rep}_baseline_synth_seed0.json",
                f"split_{rep}_baseline.json",
            ]),
        })

    for rep in REPS:
        for holdout in HOLDOUTS:
            specs.append({
                "rep": rep,
                "setting": f"holdout_{holdout}",
                "dataset_path": first_existing([
                    f"dataset_{rep}_synth_test_only_holdout_{holdout}.pt",
                    f"dataset_{rep}_test_only_holdout_{holdout}.pt",
                ]),
                "ckpt_path": first_existing([
                    f"detr_{rep}_synth_holdout_{holdout}.pt",
                    f"detr_{rep}_holdout_{holdout}.pt",
                ]),
                "split_path": None,
            })

    return specs


def run_confidence_quality_analysis():
    prediction_tables = []
    summary_rows = []

    for spec in get_eval_specs():
        rep = spec["rep"]
        setting = spec["setting"]
        dataset_path = spec["dataset_path"]
        ckpt_path = spec["ckpt_path"]
        split_path = spec["split_path"]

        print(f"\n{setting} | {rep}")
        print("dataset   :", dataset_path)
        print("checkpoint:", ckpt_path)
        print("split     :", split_path)

        if dataset_path is None or ckpt_path is None:
            print("Skipping: missing dataset or checkpoint.")
            continue

        dataset = XPSSetDataset(dataset_path)

        if split_path is not None:
            with open(split_path, "r") as f:
                split = json.load(f)
            dataset = Subset(dataset, split["val_idx"])

        model = load_model_from_ckpt(ckpt_path)
        pred_df = collect_prediction_distances(dataset, model)

        if pred_df.empty:
            print("Skipping: no predictions above confidence threshold.")
            continue

        pred_df["rep"] = rep
        pred_df["setting"] = setting
        prediction_tables.append(pred_df)

        summary_rows.append(
            summarise_best_worst(
                pred_df,
                rep=rep,
                setting=setting,
                dataset_path=dataset_path,
                ckpt_path=ckpt_path,
                split_path=split_path,
            )
        )

    prediction_df = pd.concat(prediction_tables, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    prediction_df.to_csv(OUT_DIR / "prediction_level_uncertainty.csv", index=False)
    summary_df.to_csv(OUT_DIR / "combined_baseline_holdout_confidence_summary.csv", index=False)

    print(f"\nSaved outputs in: {OUT_DIR.resolve()}")
    return prediction_df, summary_df


def plot_confidence_quality(summary_df, out_dir=OUT_DIR):
    import matplotlib.pyplot as plt

    colors = {
        "skipatom": "#9A9A9A",
        "matscholar": "#444444",
        "soap": "#2E8B57",
    }

    label_map = {
        "baseline": "Base",
        "holdout_CBr": "C–Br",
        "holdout_aromatic_C": "Arom C",
        "holdout_carbonyl_O": "C=O",
    }

    setting_order = ["Base", "C–Br", "Arom C", "C=O"]
    rep_order = ["skipatom", "matscholar", "soap"]

    rep_labels = {
        "skipatom": "SkipAtom",
        "matscholar": "MatScholar",
        "soap": "SOAP",
    }

    plot_df = summary_df.copy()
    plot_df["rep"] = plot_df["rep"].str.lower().str.strip()
    plot_df["setting_short"] = plot_df["setting"].map(label_map)

    pivot = plot_df.pivot(
        index="setting_short",
        columns="rep",
        values="delta_mean_conf_best_minus_worst",
    )

    pivot = pivot.reindex(setting_order)
    pivot = pivot[rep_order]

    fig, ax = plt.subplots(figsize=(6.5, 4))

    pivot.plot(
        kind="bar",
        ax=ax,
        color=[colors[rep] for rep in pivot.columns],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_ylabel("Δ confidence (best − worst)")
    ax.set_xlabel("")
    ax.set_ylim(-0.25, 0.2)
    ax.set_title("Confidence vs quality")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [rep_labels[label] for label in labels],
        title="Representation",
        frameon=False,
        loc="lower right",
    )

    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    save_path = out_dir / "confidenceplot.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    _, summary_df = run_confidence_quality_analysis()
    plot_confidence_quality(summary_df)