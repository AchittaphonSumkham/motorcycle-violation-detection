"""5-Fold cross-validation evaluation: mean +/- SD per model variant.

Expects weights laid out as produced by training.train --mode kfold:
    <models-root>/<variant>_Fold_<N>/weights/best.pt

Usage:
    python -m evaluation.kfold_cv_eval \
        --models-root /path/to/model_runs \
        --data /path/to/test_data.yaml --imgsz 512 \
        [--variants yolov8n yolov8s yolo11n yolo11s yolo26n yolo26s] \
        [--folds 5] [--device 0] [--out evaluation/results/kfold_results.csv] [--dry-run]
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HPARAMS = REPO_ROOT / "configs" / "hyperparameters.yaml"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "results" / "kfold_results.csv"


def load_defaults(path: Path) -> tuple:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg["model_variants"], cfg["kfold"]["k"]


def weight_path(models_root: Path, variant: str, fold: int) -> Path:
    return models_root / f"{variant}_Fold_{fold}" / "weights" / "best.pt"


def evaluate_variant(variant: str, models_root: Path, folds: range, data: Path,
                     imgsz: int, device: str) -> list:
    rows = []
    for fold in folds:
        model_path = weight_path(models_root, variant, fold)
        if not model_path.exists():
            print(f"WARNING: weights not found, skipping: {model_path}")
            continue

        print(f"Evaluating {variant} fold {fold}...")
        model = YOLO(str(model_path))
        results = model.val(data=str(data), split="val", imgsz=imgsz, device=device)

        metrics = results.results_dict
        p = metrics.get("metrics/precision(B)", 0)
        r = metrics.get("metrics/recall(B)", 0)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

        rows.append({
            "Model": variant,
            "Fold": fold,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "mAP50": metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
        })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Append MEAN and SD rows per model variant."""
    metric_cols = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]
    out_frames = [df]
    for variant, group in df.groupby("Model", sort=False):
        mean_row = {"Model": variant, "Fold": "MEAN"}
        sd_row = {"Model": variant, "Fold": "SD"}
        for col in metric_cols:
            mean_row[col] = group[col].mean()
            sd_row[col] = group[col].std(ddof=1) if len(group) > 1 else 0.0
        out_frames.append(pd.DataFrame([mean_row, sd_row]))
    return pd.concat(out_frames, ignore_index=True)


def main() -> None:
    default_variants, default_k = load_defaults(DEFAULT_HPARAMS)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-root", type=Path, required=True,
                        help="folder containing <variant>_Fold_N/weights/best.pt")
    parser.add_argument("--data", type=Path, required=True, help="test/val data.yaml")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--variants", nargs="+", default=default_variants)
    parser.add_argument("--folds", type=int, default=default_k)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="list expected weight paths and whether they exist, then exit")
    args = parser.parse_args()

    folds = range(1, args.folds + 1)

    if args.dry_run:
        for variant in args.variants:
            for fold in folds:
                p = weight_path(args.models_root, variant, fold)
                status = "OK" if p.exists() else "MISSING"
                print(f"[{status}] {p}")
        return

    all_rows = []
    for variant in args.variants:
        all_rows.extend(evaluate_variant(variant, args.models_root, folds,
                                         args.data, args.imgsz, args.device))

    if not all_rows:
        raise SystemExit("ERROR: no weights were found under --models-root")

    df = pd.DataFrame(all_rows)
    df_summary = summarize(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
