import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from scripts.dataset import XPSSetDataset, collate_set
from scripts.detr_model import SpectrumDETR


def compute_set_accuracy(
    env_preds,
    conf_logits,
    targets,
    acc_threshold=0.2,
    conf_threshold=0.5,
):
    conf = torch.sigmoid(conf_logits)
    batch_acc = []

    for pred_env, pred_conf, tgt in zip(env_preds, conf, targets):
        keep = pred_conf > conf_threshold
        pred_env = pred_env[keep]

        if len(tgt) == 0:
            batch_acc.append(1.0 if len(pred_env) == 0 else 0.0)
            continue

        if len(pred_env) == 0:
            batch_acc.append(0.0)
            continue

        pred_norm = pred_env / (pred_env.norm(dim=1, keepdim=True) + 1e-8)
        tgt_norm = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-8)

        dists = 1.0 - torch.matmul(pred_norm, tgt_norm.T)
        recovered = (dists.min(dim=0).values < acc_threshold).float()

        batch_acc.append(recovered.mean().item())

    return float(np.mean(batch_acc))


@torch.no_grad()
def evaluate_val_accuracy(
    model,
    loader,
    device,
    acc_threshold=0.2,
    conf_threshold=0.5,
):
    model.eval()
    accs = []

    for batch in loader:
        x = batch["spectra"].to(device)
        targets = [target.to(device) for target in batch["targets"]]

        env_preds, conf_logits = model(x)

        acc = compute_set_accuracy(
            env_preds,
            conf_logits,
            targets,
            acc_threshold=acc_threshold,
            conf_threshold=conf_threshold,
        )
        accs.append(acc)

    return float(np.mean(accs))


def build_model(device):
    model = SpectrumDETR(
        d_model=256,
        d_env=128,
        n_queries=24,
        n_heads=8,
        n_layers=3,
    )
    return model.to(device)


def resolve_seeded_or_legacy_path(path_with_seed, legacy_path, seed):
    path_with_seed = Path(path_with_seed)
    legacy_path = Path(legacy_path)

    if path_with_seed.exists():
        return str(path_with_seed)

    if seed == 0 and legacy_path.exists():
        return str(legacy_path)

    return None


def run_eval(
    rep,
    setting,
    seed,
    train_pt,
    split_path,
    ckpt_path,
    device,
    batch_size=8,
    acc_threshold=0.2,
    conf_threshold=0.5,
):
    print(f"\nEvaluating rep={rep} | setting={setting} | seed={seed}")

    dataset = XPSSetDataset(train_pt)

    with open(split_path, "r") as f:
        split = json.load(f)

    val_idx = split["val_idx"]
    val_dataset = Subset(dataset, val_idx)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_set,
        num_workers=0,
    )

    model = build_model(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    val_acc = evaluate_val_accuracy(
        model,
        val_loader,
        device,
        acc_threshold=acc_threshold,
        conf_threshold=conf_threshold,
    )

    print(f"  val_accuracy = {val_acc:.4f} ({100 * val_acc:.2f}%)")

    return {
        "rep": rep,
        "setting": setting,
        "seed": seed,
        "val_accuracy": val_acc,
        "n_val_examples": len(val_dataset),
        "acc_threshold": acc_threshold,
        "conf_threshold": conf_threshold,
        "checkpoint": ckpt_path,
        "split_file": split_path,
        "train_dataset": train_pt,
    }


def evaluate_all_val_accuracies(
    reps=("skipatom", "matscholar", "soap"),
    holdouts=("CBr", "carbonyl_O", "aromatic_C"),
    seeds=(0,),
    batch_size=8,
    acc_threshold=0.2,
    conf_threshold=0.5,
    device=None,
    save=True,
    runs_csv="val_accuracy_runs.csv",
    summary_csv="val_accuracy_summary.csv",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    # ----------------------------
    # BASELINE SYNTH
    # ----------------------------
    for rep in reps:
        train_pt = f"dataset_{rep}_synth.pt"
        split_path = f"split_{rep}_baseline_synth.json"
        ckpt_path = f"detr_{rep}_baseline.pt"

        if Path(train_pt).exists() and Path(split_path).exists() and Path(ckpt_path).exists():
            rows.append(
                run_eval(
                    rep=rep,
                    setting="baseline_synth",
                    seed=0,
                    train_pt=train_pt,
                    split_path=split_path,
                    ckpt_path=ckpt_path,
                    device=device,
                    batch_size=batch_size,
                    acc_threshold=acc_threshold,
                    conf_threshold=conf_threshold,
                )
            )
        else:
            print(f"Skipping rep={rep} baseline_synth (missing files)")
            
            if train_pt and split_path and ckpt_path:
                rows.append(
                    run_eval(
                        rep=rep,
                        setting="baseline_synth",
                        seed=seed,
                        train_pt=train_pt,
                        split_path=split_path,
                        ckpt_path=ckpt_path,
                        device=device,
                        batch_size=batch_size,
                        acc_threshold=acc_threshold,
                        conf_threshold=conf_threshold,
                    )
                )
            else:
                print(f"Skipping rep={rep}, seed={seed} baseline_synth (missing files)")

    # ----------------------------
    # OOD HOLDOUTS
    # ----------------------------
    for rep in reps:
        for holdout in holdouts:
            for seed in seeds:
                train_pt_seeded = f"dataset_{rep}_synth_train_no_holdout_{holdout}_seed{seed}.pt"
                split_seeded = f"split_{rep}_holdout_{holdout}_seed{seed}.json"
                ckpt_seeded = f"detr_{rep}_holdout_{holdout}_seed{seed}.pt"

                train_pt_legacy = f"dataset_{rep}_synth_train_no_holdout_{holdout}.pt"
                split_legacy = f"split_{rep}_holdout_{holdout}.json"
                ckpt_legacy = f"detr_{rep}_holdout_{holdout}.pt"

                train_pt = resolve_seeded_or_legacy_path(train_pt_seeded, train_pt_legacy, seed)
                split_path = resolve_seeded_or_legacy_path(split_seeded, split_legacy, seed)
                ckpt_path = resolve_seeded_or_legacy_path(ckpt_seeded, ckpt_legacy, seed)

                if train_pt and split_path and ckpt_path:
                    rows.append(
                        run_eval(
                            rep=rep,
                            setting=f"holdout_{holdout}",
                            seed=seed,
                            train_pt=train_pt,
                            split_path=split_path,
                            ckpt_path=ckpt_path,
                            device=device,
                            batch_size=batch_size,
                            acc_threshold=acc_threshold,
                            conf_threshold=conf_threshold,
                        )
                    )
                else:
                    print(f"Skipping rep={rep}, holdout={holdout}, seed={seed} (missing files)")
                    
    runs_df = pd.DataFrame(rows)

    if not runs_df.empty:
        runs_df = runs_df.sort_values(["setting", "rep", "seed"]).reset_index(drop=True)

        summary_df = (
            runs_df.groupby(["rep", "setting"], as_index=False)
            .agg(
                mean_val_accuracy=("val_accuracy", "mean"),
                std_val_accuracy=("val_accuracy", "std"),
                n_seeds=("seed", "count"),
                n_val_examples=("n_val_examples", "first"),
                acc_threshold=("acc_threshold", "first"),
                conf_threshold=("conf_threshold", "first"),
            )
        )

        summary_df["mean_val_accuracy_%"] = 100 * summary_df["mean_val_accuracy"]
        summary_df["std_val_accuracy_%"] = 100 * summary_df["std_val_accuracy"].fillna(0.0)
    else:
        summary_df = pd.DataFrame()

    if save:
        runs_df.to_csv(runs_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSaved: {runs_csv}")
        print(f"Saved: {summary_csv}")

    return runs_df, summary_df


if __name__ == "__main__":
    runs_df, summary_df = evaluate_all_val_accuracies()
    print(runs_df)
    print(summary_df)