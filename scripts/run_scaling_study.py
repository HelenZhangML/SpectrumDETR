import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REP = "soap"
DATASET_PATH = f"dataset_{REP}_synth.pt"
SPLIT_PATH = f"split_{REP}_baseline_synth.json"

TRAIN_FRACTIONS = [0.10, 0.25, 0.50]
SEED = 0

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4

ACC_THRESHOLD = 0.2
CONF_THRESHOLD = 0.5

OUT_CSV = f"scaling_study_{REP}.csv"
OUT_PLOT = f"scaling_study_{REP}_val_accuracy.png"


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ============================================================
# ACCURACY METRIC
# ============================================================
def compute_set_accuracy(env_preds, conf_logits, targets,
                         acc_threshold=0.2, conf_threshold=0.5):
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

        # each true target is counted as recovered if any prediction is close enough
        recovered = (dists.min(dim=0).values < acc_threshold).float()
        batch_acc.append(recovered.mean().item())

    return float(np.mean(batch_acc))


@torch.no_grad()
def evaluate_val_accuracy(model, loader,
                          acc_threshold=0.2, conf_threshold=0.5):
    model.eval()
    accs = []

    for batch in loader:
        x = batch["spectra"].to(DEVICE)
        targets = [t.to(DEVICE) for t in batch["targets"]]

        env_preds, conf_logits = model(x)
        acc = compute_set_accuracy(
            env_preds, conf_logits, targets,
            acc_threshold=acc_threshold,
            conf_threshold=conf_threshold
        )
        accs.append(acc)

    return float(np.mean(accs))


# ============================================================
# GROUP-SAFE TRAIN SUBSAMPLING
# ============================================================
def group_indices_by_key(dataset, indices):
    group_to_indices = defaultdict(list)
    for idx in indices:
        k = dataset[idx]["key"]
        group_to_indices[k].append(idx)
    return group_to_indices


def subsample_train_groups(dataset, train_idx, fraction, seed=0):
    """
    Subsample whole polymer groups from the existing training split.
    Keeps validation/test fixed and preserves no-leakage logic.
    """
    if fraction >= 1.0:
        return list(train_idx)

    rng = random.Random(seed)
    group_to_indices = group_indices_by_key(dataset, train_idx)
    groups = list(group_to_indices.keys())
    rng.shuffle(groups)

    n_keep = max(1, int(round(fraction * len(groups))))
    keep_groups = groups[:n_keep]

    new_idx = []
    for g in keep_groups:
        new_idx.extend(group_to_indices[g])

    return sorted(new_idx)


# ============================================================
# LOAD DATA + FIXED SPLIT
# ============================================================
ds = XPSSetDataset(DATASET_PATH)

with open(SPLIT_PATH, "r") as f:
    split = json.load(f)

train_idx_full = split["train_idx"]
val_idx = split["val_idx"]
test_idx = split["test_idx"]

print("Loaded split info:")
print(split.get("info", {}))
print(f"Full train: {len(train_idx_full)} | val: {len(val_idx)} | test: {len(test_idx)}")


# fixed val/test loaders
ds_val = Subset(ds, val_idx)
val_loader = DataLoader(
    ds_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_set,
    num_workers=0
)


# ============================================================
# RUN SCALING STUDY
# ============================================================
rows = []

for frac in TRAIN_FRACTIONS:
    print("\n" + "=" * 80)
    print(f"TRAIN FRACTION = {frac:.2f}")
    print("=" * 80)

    set_seed(SEED)

    train_idx_sub = subsample_train_groups(ds, train_idx_full, frac, seed=SEED)
    ds_train = Subset(ds, train_idx_sub)

    train_loader = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_set,
        num_workers=0
    )

    model = SpectrumDETR(
        d_model=256,
        d_env=128,
        n_queries=24,
        n_heads=8,
        n_layers=3,
    )

    frac_tag = str(frac).replace(".", "p")
    ckpt_path = f"detr_{REP}_baseline_synth_frac_{frac_tag}.pt"
    metrics_path = f"metrics_{REP}_baseline_synth_frac_{frac_tag}.json"

    history = train_detr(
        model,
        train_loader,
        val_loader,
        DEVICE,
        epochs=EPOCHS,
        lr=LR,
        w_emb=1.0,
        w_conf=0.5,
        use_cosine=True,
        ckpt_path=ckpt_path,
        metrics_path=metrics_path,
    )

    # reload saved checkpoint
    model = SpectrumDETR(
        d_model=256,
        d_env=128,
        n_queries=24,
        n_heads=8,
        n_layers=3,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    val_acc = evaluate_val_accuracy(
        model,
        val_loader,
        acc_threshold=ACC_THRESHOLD,
        conf_threshold=CONF_THRESHOLD
    )

    row = {
        "representation": REP,
        "train_fraction": frac,
        "n_train_examples": len(train_idx_sub),
        "n_val_examples": len(val_idx),
        "val_accuracy": val_acc,
        "val_accuracy_%": 100 * val_acc,
        "final_train_loss": history["loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": min(history["val_loss"]),
        "checkpoint": ckpt_path,
        "metrics_file": metrics_path,
    }
    rows.append(row)

    print(row)


# ============================================================
# SAVE RESULTS
# ============================================================
results_df = pd.DataFrame(rows).sort_values("train_fraction").reset_index(drop=True)
results_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved: {OUT_CSV}")
display(results_df)


# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(6, 4))
plt.plot(results_df["train_fraction"], results_df["val_accuracy_%"], marker="o")
plt.xlabel("Training fraction")
plt.ylabel("Validation accuracy (%)")
plt.title(f"{REP.upper()} scaling study")
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Saved: {OUT_PLOT}")