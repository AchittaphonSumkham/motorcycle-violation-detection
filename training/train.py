"""Train a YOLO model variant, either a single holdout run or across K-Fold folds.

Hyperparameters come from configs/hyperparameters.yaml; per-run CLI flags
override individual values without editing the YAML.

Usage:
    # 80/10/10 holdout split
    python -m training.train --model yolov8n --mode holdout \
        --data /path/dataset_80_10_10/data.yaml

    # K-Fold: one fold
    python -m training.train --model yolo11s --mode kfold --fold 3 \
        --kfold-root /path/Kfold_seed12

    # K-Fold: all folds (loops fold 1..k)
    python -m training.train --model yolo26s --mode kfold \
        --kfold-root /path/Kfold_seed12

    # preview resolved train() kwargs without training
    python -m training.train --model yolov8n --mode holdout --data ... --dry-run
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HPARAMS = REPO_ROOT / "configs" / "hyperparameters.yaml"


def load_hparams(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(cfg["train"])


def compute_summary(results) -> dict:
    metrics = results.results_dict
    p = metrics.get("metrics/precision(B)", 0)
    r = metrics.get("metrics/recall(B)", 0)
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    return {
        "Precision": p,
        "Recall": r,
        "F1-Score": f1,
        "mAP50": metrics.get("metrics/mAP50(B)", 0),
        "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
        "Fitness": results.fitness,
    }


def print_summary(title: str, summary: dict) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"{'Metric':<20} | {'Value'}")
    print("-" * 50)
    for key, val in summary.items():
        print(f"{key:<20} | {val:.4f}")
    print("=" * 50)


def free_memory(model) -> None:
    del model
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_holdout(model_variant: str, data_yaml: Path, hparams: dict, project: Path) -> None:
    if not data_yaml.exists():
        sys.exit(f"ERROR: data.yaml not found: {data_yaml}")

    project.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_variant}_run"
    print(f"\n--- Training: {run_name} ---")

    model = YOLO(f"{model_variant}.pt")
    results = model.train(data=str(data_yaml), name=run_name, project=str(project), **hparams)

    summary = compute_summary(results)
    print_summary("Training results", summary)

    summary_path = project / "training_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    free_memory(model)


def run_kfold(model_variant: str, kfold_root: Path, hparams: dict, project: Path,
             folds: list) -> None:
    project.mkdir(parents=True, exist_ok=True)
    summary_results = []

    for fold_num in folds:
        yaml_path = kfold_root / f"data_fold_{fold_num}.yaml"
        if not yaml_path.exists():
            print(f"WARNING: fold file not found, skipping: {yaml_path}")
            continue

        print(f"\n--- Training fold {fold_num} ---")
        model = YOLO(f"{model_variant}.pt")
        results = model.train(
            data=str(yaml_path),
            name=f"{model_variant}_Fold_{fold_num}",
            project=str(project),
            **hparams,
        )

        summary = compute_summary(results)
        summary["Fold"] = fold_num
        summary_results.append(summary)
        print(f"Fold {fold_num} done: mAP50={summary['mAP50']:.4f}")

        free_memory(model)

    if not summary_results:
        sys.exit("ERROR: no folds were trained (check --kfold-root / --fold)")

    df_summary = pd.DataFrame(summary_results)
    summary_path = project / "kfold_training_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("K-Fold cross-validation summary")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("-" * 60)
    for col in ["Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"]:
        print(f"{col:<20} | {df_summary[col].mean():.4f}")
    print(f"\nSaved: {summary_path}")


def resolve_folds(fold_arg, k: int) -> list:
    if fold_arg is not None:
        return [fold_arg]
    return list(range(1, k + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True,
                        help="ultralytics model name, e.g. yolov8n, yolo11s, yolo26s")
    parser.add_argument("--mode", choices=["holdout", "kfold"], required=True)
    parser.add_argument("--data", type=Path, help="data.yaml (required for --mode holdout)")
    parser.add_argument("--kfold-root", type=Path,
                        help="folder containing data_fold_N.yaml (required for --mode kfold)")
    parser.add_argument("--fold", type=int, help="train a single fold (kfold mode only)")
    parser.add_argument("--hparams", type=Path, default=DEFAULT_HPARAMS,
                        help="hyperparameters YAML (default: configs/hyperparameters.yaml)")
    parser.add_argument("--project", type=Path, default=Path("runs_out"),
                        help="ultralytics project (output) folder")
    parser.add_argument("--device", help="override device, e.g. 0 or cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved model.train() kwargs and exit")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.hparams.read_text(encoding="utf-8"))
    hparams = dict(cfg["train"])
    k = cfg["kfold"]["k"]

    for key, val in [("device", args.device), ("epochs", args.epochs),
                      ("batch", args.batch), ("imgsz", args.imgsz)]:
        if val is not None:
            hparams[key] = val

    if args.mode == "holdout" and args.data is None:
        parser.error("--data is required for --mode holdout")
    if args.mode == "kfold" and args.kfold_root is None:
        parser.error("--kfold-root is required for --mode kfold")

    if args.dry_run:
        print(f"model     = {args.model}")
        print(f"mode      = {args.mode}")
        print(f"project   = {args.project}")
        if args.mode == "holdout":
            print(f"data      = {args.data}")
        else:
            print(f"kfold_root= {args.kfold_root}")
            print(f"folds     = {resolve_folds(args.fold, k)}")
        print("train() kwargs:")
        for key, val in hparams.items():
            print(f"  {key} = {val!r}")
        return

    if args.mode == "holdout":
        run_holdout(args.model, args.data, hparams, args.project)
    else:
        run_kfold(args.model, args.kfold_root, hparams, args.project,
                 resolve_folds(args.fold, k))


if __name__ == "__main__":
    main()
