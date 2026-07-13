"""Count class instances across YOLO label files (dataset QA helper).

Usage:
    python -m data_preparation.dataset_stats --labels /path/to/dataset/train/labels
"""

import argparse
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "hyperparameters.yaml"


def load_class_names(config_path: Path) -> dict:
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in cfg["classes"].items()}
    except (OSError, KeyError, TypeError, ValueError):
        return {}


def count_instances(label_dir: Path) -> tuple:
    counts = {}
    file_count = 0
    for txt_file in label_dir.glob("*.txt"):
        file_count += 1
        for line in txt_file.read_text().splitlines():
            parts = line.split()
            if parts:
                class_id = int(parts[0])
                counts[class_id] = counts.get(class_id, 0) + 1
    return counts, file_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True,
                        help="folder containing YOLO .txt label files")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CONFIG,
                        help="YAML with a 'classes' id->name map (default: configs/hyperparameters.yaml)")
    args = parser.parse_args()

    if not args.labels.exists():
        sys.exit(f"ERROR: folder not found: {args.labels}")

    names = load_class_names(args.classes)
    counts, file_count = count_instances(args.labels)

    print(f"Instance counts from {file_count} label files:")
    print("-" * 40)
    for cid in sorted(counts):
        label = names.get(cid, "?")
        print(f"Class {cid} ({label}): {counts[cid]} instances")
    print("-" * 40)
    print(f"Total: {sum(counts.values())} instances")


if __name__ == "__main__":
    main()
