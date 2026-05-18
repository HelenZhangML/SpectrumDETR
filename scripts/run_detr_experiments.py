
# ============================================================
# RUNNING THE DETR EXPERIMENTS
# ============================================================
from scripts.detr_model import (
    SpectrumDETR,
    train_detr,
    evaluate_detr,
)

def run_all_experiments():
    all_results = []

    # ---------- baseline ----------
    if RUN_BASELINE:
        for rep in REPS:
            train_pt, train_meta, ckpt_path, metrics_path, split_path, results_path = get_baseline_paths(rep)

            res = run_single_experiment(
                rep=rep,
                experiment_name="baseline",
                train_pt=train_pt,
                ckpt_path=ckpt_path,
                metrics_path=metrics_path,
                split_path=split_path,
                results_path=results_path,
                ood_pt=None,
                holdout_name=None,
            )
            if res is not None:
                all_results.append(res)

    # ---------- OOD holdouts ----------
    if RUN_OOD:
        for rep in REPS:
            for holdout_name in HOLDOUTS:
                (
                    train_pt,
                    train_meta,
                    ood_pt,
                    ood_meta,
                    ckpt_path,
                    metrics_path,
                    split_path,
                    results_path,
                ) = get_holdout_paths(rep, holdout_name)

                res = run_single_experiment(
                    rep=rep,
                    experiment_name="holdout",
                    train_pt=train_pt,
                    ckpt_path=ckpt_path,
                    metrics_path=metrics_path,
                    split_path=split_path,
                    results_path=results_path,
                    ood_pt=ood_pt,
                    holdout_name=holdout_name,
                )
                if res is not None:
                    all_results.append(res)

    with open(RESULTS_SUMMARY_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved master summary:", RESULTS_SUMMARY_PATH)
    print("Total completed experiments:", len(all_results))


# ============================================================
# Resume (missing only) - checks for missing metrics files and runs those experiments
# ============================================================

def find_missing_experiments():
    missing_runs = []

    if RUN_BASELINE:
        for rep in REPS:
            _, _, _, metrics_path, _, _ = get_baseline_paths(rep)
            if not Path(metrics_path).exists():
                missing_runs.append(("baseline", rep, None))

    if RUN_OOD:
        for rep in REPS:
            for holdout in HOLDOUTS:
                _, _, _, _, _, metrics_path, _, _ = get_holdout_paths(rep, holdout)
                if not Path(metrics_path).exists():
                    missing_runs.append(("holdout", rep, holdout))

    return missing_runs


def run_missing_experiments():
    missing = find_missing_experiments()

    print("\nMissing experiments:")
    for exp, rep, holdout in missing:
        print(f"{exp:8} | {rep:10} | {holdout}")

    print(f"\nTotal missing: {len(missing)}")

    for exp, rep, holdout in missing:

        print("\n" + "=" * 80)
        print(f"RUNNING: {exp} | {rep} | {holdout}")
        print("=" * 80)

        if exp == "baseline":
            train_pt, train_meta, ckpt_path, metrics_path, split_path, results_path = get_baseline_paths(rep)

            run_single_experiment(
                rep=rep,
                experiment_name="baseline",
                train_pt=train_pt,
                ckpt_path=ckpt_path,
                metrics_path=metrics_path,
                split_path=split_path,
                results_path=results_path,
                ood_pt=None,
                holdout_name=None,
            )

        else:
            (
                train_pt,
                train_meta,
                ood_pt,
                ood_meta,
                ckpt_path,
                metrics_path,
                split_path,
                results_path,
            ) = get_holdout_paths(rep, holdout)

            run_single_experiment(
                rep=rep,
                experiment_name="holdout",
                train_pt=train_pt,
                ckpt_path=ckpt_path,
                metrics_path=metrics_path,
                split_path=split_path,
                results_path=results_path,
                ood_pt=ood_pt,
                holdout_name=holdout,
            )

    print("\nFinished running missing experiments.")


# ============================================================
# Entry point - choose to run all experiments or just missing ones
# ============================================================

if __name__ == "__main__":
    RUN_MODE = "missing"  # "all" or "missing"

    if RUN_MODE == "all":
        run_all_experiments()
    elif RUN_MODE == "missing":
        run_missing_experiments()