"""Multilabel Stratified K-Fold splitter for the motorcycle-violation dataset.

Each image is described by the feature vector
    [is_day, is_night, motorcycle_count, with_helmet_count, no_helmet_count]
(day/night comes from the 'D'/'N' filename prefix) and distributed across K
folds with MultilabelStratifiedKFold, so every fold preserves both the
lighting-condition ratio and the per-class instance distribution.

Usage:
    # split (copies images/labels into <output>/Fold_N/{train,val})
    python -m data_preparation.stratified_kfold_split \
        --train-root /path/to/dataset/train --output /path/to/Kfold_seed12

    # preview fold distributions without copying anything
    ... --dry-run

    # check an existing split: every image must appear K-1 times in train, once in val
    python -m data_preparation.stratified_kfold_split --output /path/to/Kfold_seed12 --verify-only
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "hyperparameters.yaml"


def load_kfold_defaults(config_path: Path) -> tuple:
    """Read (k, seed) from configs/hyperparameters.yaml; fall back to (5, 12)."""
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return int(cfg["kfold"]["k"]), int(cfg["kfold"]["seed"])
    except (OSError, KeyError, TypeError, ValueError):
        return 5, 12


def get_label_stats(label_path: Path) -> dict:
    """Count instances per class id in one YOLO label file."""
    counts = {0: 0, 1: 0, 2: 0}
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if parts:
                class_id = int(parts[0])
                if class_id in counts:
                    counts[class_id] += 1
    return counts


def build_feature_matrix(images: np.ndarray, lbl_dir: Path) -> np.ndarray:
    """Feature vector per image: [is_day, is_night, motorcycle, with_helmet, no_helmet]."""
    rows = []
    for img in images:
        prefix = img[0].upper()
        stats = get_label_stats(lbl_dir / (Path(img).stem + ".txt"))
        rows.append([
            1 if prefix == "D" else 0,
            1 if prefix == "N" else 0,
            stats[0], stats[1], stats[2],
        ])
    return np.array(rows)


def print_fold_summary(fold_num: int, images: np.ndarray, Y: np.ndarray,
                       train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    for split_type, indices in [("train", train_idx), ("val", val_idx)]:
        sub = Y[indices]
        print(
            f"  Fold {fold_num} {split_type:5s}: {len(indices):5d} images | "
            f"day {int(sub[:, 0].sum()):5d} / night {int(sub[:, 1].sum()):5d} | "
            f"motorcycle {int(sub[:, 2].sum()):6d}, with_helmet {int(sub[:, 3].sum()):6d}, "
            f"no_helmet {int(sub[:, 4].sum()):6d}"
        )


def run_multilabel_kfold(train_root: Path, output_root: Path, k: int, seed: int,
                         dry_run: bool) -> None:
    img_dir = train_root / "images"
    lbl_dir = train_root / "labels"
    if not img_dir.exists():
        sys.exit(f"ERROR: images folder not found: {img_dir}")

    images = np.array(sorted(
        f.name for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
    ))
    print(f"Found {len(images)} images. Building multilabel feature matrix...")
    Y = build_feature_matrix(images, lbl_dir)

    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    log_data = []

    for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(images, Y)):
        fold_num = fold_idx + 1
        print(f"\nFold {fold_num}/{k}:")
        print_fold_summary(fold_num, images, Y, train_idx, val_idx)
        if dry_run:
            continue

        fold_path = output_root / f"Fold_{fold_num}"
        for split_type, indices in [("train", train_idx), ("val", val_idx)]:
            target_img_dir = fold_path / split_type / "images"
            target_lbl_dir = fold_path / split_type / "labels"
            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_lbl_dir.mkdir(parents=True, exist_ok=True)

            for idx in indices:
                img_name = images[idx]
                lbl_name = Path(img_name).stem + ".txt"
                features = Y[idx]

                shutil.copy2(img_dir / img_name, target_img_dir / img_name)
                src_lbl = lbl_dir / lbl_name
                if src_lbl.exists():
                    shutil.copy2(src_lbl, target_lbl_dir / lbl_name)

                log_data.append({
                    "Fold": fold_num,
                    "Type": split_type,
                    "Filename": img_name,
                    "Light": "Day" if features[0] == 1 else "Night",
                    "Motorcycle_Count": features[2],
                    "Helmet_Count": features[3],
                    "No_Helmet_Count": features[4],
                })

        yaml_content = f"""path: {fold_path.resolve()}
train: train/images
val: val/images

names:
  0: motorcycle
  1: with_helmet
  2: no_helmet
"""
        (output_root / f"data_fold_{fold_num}.yaml").write_text(yaml_content, encoding="utf-8")

    if dry_run:
        print("\nDry run: nothing copied.")
        return

    pd.DataFrame(log_data).to_csv(output_root / "kfold_multilabel_log.csv", index=False)
    print(f"\nDone. K-Fold split written to: {output_root}")


def verify_kfold_integrity(k_root: Path, k: int) -> bool:
    """Every image must appear K-1 times in train and exactly once in val."""
    file_stats = {}
    print(f"Verifying {k}-Fold integrity under: {k_root}")

    for fold_idx in range(1, k + 1):
        for split_type in ("train", "val"):
            img_dir = k_root / f"Fold_{fold_idx}" / split_type / "images"
            if not img_dir.exists():
                print(f"ERROR: folder not found: {img_dir}")
                continue
            for f in img_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    stats = file_stats.setdefault(f.name, {"train": 0, "val": 0})
                    stats[split_type] += 1

    errors = [
        f"  {fname}: train={c['train']}, val={c['val']}"
        for fname, c in file_stats.items()
        if c["train"] != k - 1 or c["val"] != 1
    ]

    print("-" * 50)
    print(f"Checked {len(file_stats)} unique images.")
    if not errors:
        print(f"OK: every image appears {k - 1}x in train and 1x in val.")
    else:
        print(f"FAILED: {len(errors)} images violate the {k - 1}/1 rule:")
        for err in errors[:10]:
            print(err)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    print("-" * 50)
    return not errors


def main() -> None:
    default_k, default_seed = load_kfold_defaults(DEFAULT_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-root", type=Path,
                        help="dataset train root containing images/ and labels/")
    parser.add_argument("--output", type=Path, required=True,
                        help="output root for Fold_N folders (or existing root for --verify-only)")
    parser.add_argument("--k", type=int, default=default_k)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--dry-run", action="store_true",
                        help="print fold distributions without copying files")
    parser.add_argument("--verify-only", action="store_true",
                        help="only run the integrity check on an existing split at --output")
    args = parser.parse_args()

    if args.verify_only:
        ok = verify_kfold_integrity(args.output, args.k)
        sys.exit(0 if ok else 1)

    if args.train_root is None:
        parser.error("--train-root is required unless --verify-only is given")
    run_multilabel_kfold(args.train_root, args.output, args.k, args.seed, args.dry_run)


if __name__ == "__main__":
    main()
