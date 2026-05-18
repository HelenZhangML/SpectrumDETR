"""
Plot training, validation, and optional OOD loss curves from metrics JSON files.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_KEYS = ["train_loss", "loss", "training_loss"]
VAL_KEYS = ["val_loss", "valid_loss", "validation_loss"]
OOD_KEYS = ["ood_loss", "test_ood_loss", "ood_val_loss", "out_of_domain_loss"]
EPOCH_KEYS = ["epoch", "epochs", "step", "steps"]


def normalise_history(metrics_obj):
    if isinstance(metrics_obj, dict):
        return metrics_obj

    if isinstance(metrics_obj, list):
        history = {}
        for row in metrics_obj:
            for key, value in row.items():
                history.setdefault(key, []).append(value)
        return history

    raise ValueError(f"Unsupported metrics format: {type(metrics_obj)}")


def pick_key(history, candidates):
    for key in candidates:
        if key in history:
            return key
    return None

def plot_loss_curves(metrics_dir=".", out_dir="loss_curve_plots", show=False):
    metrics_dir = Path(metrics_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    metrics_files = sorted(metrics_dir.glob("metrics_*.json"))

    print(f"Found {len(metrics_files)} metrics files")
    for path in metrics_files:
        print(" ", path.name)

    for path in metrics_files:
        with open(path, "r") as f:
            history = normalise_history(json.load(f))

        train_key = pick_key(history, TRAIN_KEYS)
        val_key = pick_key(history, VAL_KEYS)
        ood_key = pick_key(history, OOD_KEYS)
        x_key = pick_key(history, EPOCH_KEYS)

        if train_key is None and val_key is None and ood_key is None:
            print(f"Skipping {path.name} (no recognised loss keys)")
            continue

        if x_key is not None:
            x = history[x_key]
        else:
            first_key = train_key or val_key or ood_key
            x = list(range(1, len(history[first_key]) + 1))

        plt.figure(figsize=(7, 5))

        if train_key is not None:
            plt.plot(x[:len(history[train_key])], history[train_key], label="Train")

        if val_key is not None:
            plt.plot(x[:len(history[val_key])], history[val_key], label="Validation")

        if ood_key is not None:
            plt.plot(x[:len(history[ood_key])], history[ood_key], label="OOD")

        plt.xlabel("Epoch" if x_key in ["epoch", "epochs"] else "Step")
        plt.ylabel("Loss")
        plt.title(path.stem.replace("metrics_", "").replace("_", " "))
        plt.legend()
        plt.tight_layout()

        save_path = out_dir / f"{path.stem}_losses.png"
        plt.savefig(save_path, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()
        print(f"Saved → {save_path}")
        

# Creates the figure used in the report
def plot_appendix_loss_grid(
    metrics_dir=".",
    out_dir="loss_curve_plots",
    show=False,
):
    """
    Compact appendix figures showing all regimes together.

    For each loss type:
        rows = representations
        cols = regimes

    Outputs:
        appendix_total_loss_grid.png
        appendix_confidence_loss_grid.png
        appendix_embedding_loss_grid.png
    """

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    metrics_dir = Path(metrics_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    reps = ["skipatom", "matscholar", "soap"]

    rep_labels = {
        "skipatom": "SkipAtom",
        "matscholar": "MatScholar",
        "soap": "SOAP",
    }

    regimes = [
        ("full", "Full"),
        ("cbr", "C--Br"),
        ("aromatic", "Aromatic C"),
        ("carbonyl", "Carbonyl O"),
    ]

    loss_types = [
        ("total", "loss", "val_loss", "Total loss"),
        ("confidence", "conf", "val_conf", "Confidence loss"),
        ("embedding", "emb", "val_emb", "Embedding loss"),
    ]

    metrics_files = sorted(metrics_dir.glob("metrics_*.json"))

    def find_file(rep, regime):
        for p in metrics_files:
            name = p.stem.lower()

            if rep not in name:
                continue

            if regime == "full":
                if "holdout" not in name:
                    return p

            elif regime == "cbr":
                if "cbr" in name:
                    return p

            elif regime == "aromatic":
                if "aromatic" in name:
                    return p

            elif regime == "carbonyl":
                if "carbonyl" in name:
                    return p

        return None

    for loss_name, train_key, val_key, loss_label in loss_types:

        fig, axes = plt.subplots(
            nrows=len(reps),
            ncols=len(regimes),
            figsize=(14, 8),
            sharex=True,
        )

        for row_idx, rep in enumerate(reps):
            for col_idx, (regime_key, regime_label) in enumerate(regimes):

                ax = axes[row_idx, col_idx]
                p = find_file(rep, regime_key)

                if p is None:
                    ax.text(
                        0.5,
                        0.5,
                        "Missing",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                    continue

                with open(p, "r") as f:
                    history = normalise_history(json.load(f))

                has_train = train_key in history
                has_val = val_key in history

                if not has_train and not has_val:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                    continue

                if has_train:
                    x_train = range(1, len(history[train_key]) + 1)
                    ax.plot(
                        x_train,
                        history[train_key],
                        linewidth=2,
                        label="Train",
                    )

                if has_val:
                    x_val = range(1, len(history[val_key]) + 1)
                    ax.plot(
                        x_val,
                        history[val_key],
                        linewidth=2,
                        linestyle="--",
                        label="Validation",
                    )

                if row_idx == 0:
                    ax.set_title(regime_label)

                if col_idx == 0:
                    ax.set_ylabel(rep_labels[rep])

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.margins(x=0.02)

        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
        )

        fig.supxlabel("Epoch")
        fig.supylabel(loss_label)

        plt.tight_layout()

        save_path = out_dir / f"appendix_{loss_name}_loss_grid.png"

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        if show:
            plt.show()
        else:
            plt.close()

        print(f"Saved → {save_path}")
        
if __name__ == "__main__":
    plot_loss_curves()

