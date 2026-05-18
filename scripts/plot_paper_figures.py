from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts.plot_ood_pca import (
    get_combo_paths,
    load_env_matrix_from_raw_pt,
    make_loader,
    load_model_from_ckpt,
    extract_pred_env_vectors,
    build_training_subset_masks,
    MAX_TRAIN_POINTS_FOR_PLOT,
    DEVICE,
    BATCH_SIZE,
    CONF_THRESH,
)


def plot_paper_carbonyl_figure(
    reps=("skipatom", "matscholar", "soap"),
    holdout="carbonyl_O",
    out_dir=Path("ood_pca_plots_with_subsets"),
    tick_fs=14,
    legend_fs=16,
    title_fs=16,
    conf_thresh=CONF_THRESH,
    device=DEVICE,
    batch_size=BATCH_SIZE,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, len(reps), figsize=(15, 10))

    for j, rep in enumerate(reps):
        print(f"\n=== {rep} | {holdout} ===")

        paths = get_combo_paths(rep, holdout)

        required = ["train_pt", "ood_pt", "ckpt", "train_csv"]
        if any(paths[k] is None for k in required):
            raise FileNotFoundError(f"Missing required files for {rep} | {holdout}: {paths}")

        X_train_128 = load_env_matrix_from_raw_pt(paths["train_pt"])
        X_ood_true_128 = load_env_matrix_from_raw_pt(paths["ood_pt"])

        ood_loader = make_loader(paths["ood_pt"], batch_size=batch_size)
        model = load_model_from_ckpt(paths["ckpt"], device=device)

        X_pred_128 = extract_pred_env_vectors(
            model,
            ood_loader,
            device,
            conf_thresh=conf_thresh,
        )

        print("Train:", X_train_128.shape, "OOD:", X_ood_true_128.shape, "Pred:", X_pred_128.shape)

        X_train_bg = X_train_128
        if len(X_train_bg) > MAX_TRAIN_POINTS_FOR_PLOT:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X_train_bg), MAX_TRAIN_POINTS_FOR_PLOT, replace=False)
            X_train_bg = X_train_bg[idx]

        scaler = StandardScaler().fit(X_train_bg)
        pca = PCA(n_components=2, random_state=0).fit(scaler.transform(X_train_bg))

        Z_train_bg = pca.transform(scaler.transform(X_train_bg))
        Z_train_full = pca.transform(scaler.transform(X_train_128))
        Z_ood = pca.transform(scaler.transform(X_ood_true_128))
        Z_pred = (
            pca.transform(scaler.transform(X_pred_128))
            if len(X_pred_128)
            else np.empty((0, 2))
        )

        merged_df, subset_masks = build_training_subset_masks(paths["train_csv"], holdout)

        cleaned_subset_masks = {}
        for label, mask in subset_masks.items():
            cleaned_subset_masks["seen O (non-carbonyl)" if label == "other O" else label] = mask

        Z_ref = np.vstack([Z_train_bg, Z_ood])
        x_lo, x_hi = np.percentile(Z_ref[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(Z_ref[:, 1], [1, 99])
        pad_x = 0.10 * (x_hi - x_lo + 1e-12)
        pad_y = 0.10 * (y_hi - y_lo + 1e-12)

        def draw(ax, red_size=42):
            ax.scatter(
                Z_train_bg[:, 0], Z_train_bg[:, 1],
                s=8, alpha=0.10, color="grey", label="Training envs"
            )

            for label, mask in cleaned_subset_masks.items():
                if mask.sum() == 0:
                    continue
                Z_sub = Z_train_full[mask]
                ax.scatter(
                    Z_sub[:, 0], Z_sub[:, 1],
                    s=20, alpha=0.9, color="blue", label=label
                )

            ax.scatter(
                Z_ood[:, 0], Z_ood[:, 1],
                s=25, color="green", label="True OOD"
            )

            if len(Z_pred):
                ax.scatter(
                    Z_pred[:, 0], Z_pred[:, 1],
                    s=red_size, color="red", label="Predicted OOD"
                )

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=tick_fs)

        draw(axes[0, j], red_size=42)
        axes[0, j].set_title(f"{rep.capitalize()}: full view", fontsize=title_fs, pad=10)

        draw(axes[1, j], red_size=42)
        axes[1, j].set_title(f"{rep.capitalize()}: zoomed view", fontsize=title_fs, pad=10)
        axes[1, j].set_xlim(x_lo - pad_x, x_hi + pad_x)
        axes[1, j].set_ylim(y_lo - pad_y, y_hi + pad_y)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Training envs",
               markerfacecolor="0.5", markersize=8, alpha=1.0),
        Line2D([0], [0], marker="o", color="w", label="seen O (non-carbonyl)",
               markerfacecolor="blue", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="True OOD",
               markerfacecolor="green", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Predicted OOD",
               markerfacecolor="red", markersize=8),
    ]

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=legend_fs,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.30, hspace=0.30)

    save_path = out_dir / "paper_carbonyl_O_pca_2x3.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


    plt.show()

    return save_path