"""Hold-out test-set evaluation: Precision/Recall/F1/mAP50 per model.

Usage:
    python -m evaluation.holdout_test_eval \
        --weights /path/a/best.pt /path/b/best.pt \
        --data /path/dataset_80_10_10/data.yaml --imgsz 512 \
        [--device 0] [--out evaluation/results/holdout_results.csv]

    # or point at a folder of <name>/weights/best.pt runs:
    python -m evaluation.holdout_test_eval --models-root /path/to/runs --data ...
"""

import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "evaluation" / "results" / "holdout_results.csv"


def discover_weights(models_root: Path) -> list:
    """<models-root>/<name>/weights/best.pt for every subfolder that has one."""
    found = []
    for sub in sorted(models_root.iterdir()):
        candidate = sub / "weights" / "best.pt"
        if candidate.exists():
            found.append(candidate)
    return found


def evaluate(weights: list, data: Path, imgsz: int, batch: int, device: str) -> list:
    rows = []
    for model_path in weights:
        model_name = model_path.parent.parent.name
        print(f"Evaluating {model_name}...")

        model = YOLO(str(model_path))
        results = model.val(
            data=str(data), split="test", imgsz=imgsz, batch=batch,
            device=device, name=f"Test_{model_name}", exist_ok=True, plots=True,
        )

        p, r = results.box.mp, results.box.mr
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
        speed_ms = results.speed["inference"]

        rows.append({
            "Model": model_name,
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "Speed (ms)": speed_ms,
            "FPS": 1000 / speed_ms if speed_ms > 0 else 0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", nargs="+", type=Path, help="explicit list of best.pt paths")
    src.add_argument("--models-root", type=Path,
                     help="folder of <name>/weights/best.pt run directories")
    parser.add_argument("--data", type=Path, required=True, help="data.yaml with a 'test' split")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    weights = args.weights if args.weights else discover_weights(args.models_root)
    if not weights:
        raise SystemExit("ERROR: no best.pt weights found")

    rows = evaluate(weights, args.data, args.imgsz, args.batch, args.device)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
