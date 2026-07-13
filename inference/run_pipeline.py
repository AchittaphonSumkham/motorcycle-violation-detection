"""Motorcycle-violation detection pipeline: detect -> track -> classify violations.

Processes one video or a folder of videos with YOLO + ByteTrack, flags
sidewalk-intrusion (ROI test) and no-helmet violations, and writes per video:
    <output>/<video_stem>/output.mp4          annotated video
    <output>/<video_stem>/snapshots/*.jpg     cropped violation snapshots
    <output>/<video_stem>/violation_report.csv

Usage:
    python -m inference.run_pipeline \
        --model /path/to/best.pt \
        --video-dir /path/to/videos --output outputs/run1 \
        --roi ll                     # location key in configs/roi.yaml
    # or --roi-file my_roi.txt (raw YOLO polygon lines)
    # omit --roi/--roi-file to run helmet-only (sidewalk detection disabled)
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from detection.sidewalk_detection import (
    check_roi_multipoint,
    draw_roi_overlay,
    load_roi,
    parse_yolo_roi,
    scale_polygons,
)
from detection.no_helmet_detection import (
    match_helmets_to_rider,
    normalize_cls,
    split_detections,
)
from inference.violation_tracker import ViolationTracker

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference.yaml"
DEFAULT_ROI_YAML = REPO_ROOT / "configs" / "roi.yaml"


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def scan_videos(folder: Path, extensions) -> list:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def resize_to_fhd(frame, target: tuple = (1920, 1080)):
    h_src, w_src = frame.shape[:2]
    if w_src == target[0] and h_src == target[1]:
        return frame
    interp = cv2.INTER_AREA if (w_src * h_src) > (target[0] * target[1]) else cv2.INTER_LINEAR
    return cv2.resize(frame, target, interpolation=interp)


def print_device_info(device: str, frame_skip: int):
    print("-" * 60)
    print(f"Device    : {device.upper()}")
    if device.startswith("cuda") and TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  GPU     : {gpu_name}")
        print(f"  VRAM    : {vram:.1f} GB")
    print("Tracker   : ByteTrack")
    print(f"FrameSkip : {frame_skip}")
    print("-" * 60)


def process_video(video_path: Path, video_output_folder: Path, model,
                  cfg: dict, roi_polygons: list, write_video: bool = True) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ERROR: cannot open {video_path}")
        return False

    video_output_folder.mkdir(parents=True, exist_ok=True)

    target_size = tuple(cfg["frame"]["target_size"])
    frame_skip = cfg["frame"]["frame_skip"]
    conf_threshold = cfg["model"]["conf_threshold"]
    tracker_cfg = cfg["tracker"]
    sidewalk = cfg["sidewalk"]
    no_helmet_classes = {normalize_cls(c) for c in cfg["helmet"]["no_helmet_classes"]}

    v_tracker = ViolationTracker(
        output_folder=str(video_output_folder),
        sidewalk_threshold=sidewalk["confirm_threshold"],
        stationary_threshold=cfg["motion"]["stationary_threshold"],
        tolerance=cfg["motion"]["position_tolerance"],
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("    WARNING: cannot read FPS, defaulting to 30.0")
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    total_skipped = 0

    out = None
    if write_video:
        out = cv2.VideoWriter(
            str(video_output_folder / "output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            target_size,
        )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        video_seconds = frame_count / fps
        video_time_str = time.strftime("%H:%M:%S", time.gmtime(video_seconds))
        video_time_file = time.strftime("%H%M%S", time.gmtime(video_seconds))

        frame = resize_to_fhd(frame, target_size)

        if frame_count % 100 == 0 or frame_count == 1:
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0.0
            print(
                f"\r    {pct:5.1f}%  ({frame_count}/{total_frames})"
                f"  skip(no-id):{total_skipped}",
                end="", flush=True,
            )

        # frame skipping: (frame_count - 1) % frame_skip -> frame 1 always processed
        if frame_skip > 1 and (frame_count - 1) % frame_skip != 0:
            if out is not None:
                cv2.putText(frame, f"Video Time: {video_time_str}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                draw_roi_overlay(frame, roi_polygons, color=(0, 255, 255))
                out.write(frame)
            continue

        results = model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            conf=conf_threshold,
            verbose=False,
        )

        if results[0].boxes is None:
            if out is not None:
                out.write(frame)
            continue

        riders, helmets, skipped_no_id = split_detections(results[0].boxes, model.names)
        total_skipped += skipped_no_id

        v_tracker.cleanup_old_tracks({tid for _, tid in riders})
        matched_helmet_indices = set()

        for r_box, r_id in riders:
            rx1, ry1, rx2, ry2 = map(int, r_box)

            is_in_roi = check_roi_multipoint(
                rx1, ry1, rx2, ry2, roi_polygons,
                ratio=sidewalk["roi_check_ratio"],
                n_points=sidewalk["roi_check_n_points"],
                require=sidewalk["roi_check_require"],
            )
            tracking_center = (float(rx1 + rx2) / 2.0, float(ry1 + ry2) / 2.0)

            v_tracker.update_vehicle(r_id, r_box, is_in_roi, tracking_center)

            matched = match_helmets_to_rider(r_box, helmets, matched_helmet_indices)
            is_any_no_helmet = any(
                normalize_cls(h_cls) in no_helmet_classes for (_, h_cls, _) in matched
            )

            for h_box, h_cls, h_conf in matched:
                hx1, hy1, hx2, hy2 = map(int, h_box)
                is_no_h = normalize_cls(h_cls) in no_helmet_classes
                h_color = (0, 0, 255) if is_no_h else (0, 255, 0)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), h_color, 2)
                cv2.putText(frame, h_cls, (hx1, hy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 2)

            current_status = v_tracker.check_and_log(
                r_id, frame, video_time_str, video_time_file,
                is_any_no_helmet, frame_count,
            )

            color = (0, 255, 0)
            if "Sidewalk" in current_status:
                color = (0, 0, 255)
            elif "Stationary" in current_status:
                color = (0, 255, 255)

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 3)

            label = f"ID:{r_id} | {current_status}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame,
                          (rx1, ry1 - 20 - text_h),
                          (rx1 + text_w, ry1),
                          color, -1)
            cv2.putText(frame, label, (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0) if color == (0, 255, 255) else (255, 255, 255), 2)

        # unmatched helmet-class boxes, drawn raw
        for h_idx, (h_box, h_cls, h_conf) in enumerate(helmets):
            if h_idx not in matched_helmet_indices:
                hx1, hy1, hx2, hy2 = map(int, h_box)
                is_no_h = normalize_cls(h_cls) in no_helmet_classes
                h_color = (0, 0, 255) if is_no_h else (0, 255, 0)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), h_color, 2)
                cv2.putText(frame, f"{h_cls} (Raw)", (hx1, hy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 2)

        if out is not None:
            draw_roi_overlay(frame, roi_polygons, color=(0, 165, 255))
            cv2.putText(frame, f"Video Time: {video_time_str}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)

    print(f"\r    Done: {frame_count}/{total_frames} frames"
          f"  | skipped(no-id): {total_skipped}          ")

    cap.release()
    if out is not None:
        out.release()
    v_tracker.export_report(str(video_output_folder / "violation_report.csv"))
    return True


def resolve_roi(args, target_size: tuple) -> list:
    if args.roi and args.roi_file:
        sys.exit("ERROR: pass either --roi or --roi-file, not both")
    if args.roi:
        return load_roi(args.roi_yaml, args.roi, target_size)
    if args.roi_file:
        return scale_polygons(parse_yolo_roi(Path(args.roi_file).read_text()), target_size)
    print("NOTE: no --roi/--roi-file given -> sidewalk detection disabled (helmet-only mode)")
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="YOLO weights (.pt)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="single video file")
    src.add_argument("--video-dir", type=Path, help="folder of videos")
    parser.add_argument("--output", type=Path, required=True, help="output root folder")
    parser.add_argument("--roi", help="location key in configs/roi.yaml")
    parser.add_argument("--roi-file", type=Path, help="file with raw YOLO polygon lines")
    parser.add_argument("--roi-yaml", type=Path, default=DEFAULT_ROI_YAML)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="pipeline config (default: configs/inference.yaml)")
    parser.add_argument("--conf", type=float, help="override detection confidence")
    parser.add_argument("--frame-skip", type=int, help="override frame skip")
    parser.add_argument("--device", help="cuda / cpu (default: auto)")
    parser.add_argument("--no-video", action="store_true",
                        help="skip writing annotated output.mp4 (report + snapshots only)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.conf is not None:
        cfg["model"]["conf_threshold"] = args.conf
    if args.frame_skip is not None:
        cfg["frame"]["frame_skip"] = max(1, args.frame_skip)

    target_size = tuple(cfg["frame"]["target_size"])
    roi_polygons = resolve_roi(args, target_size)

    if args.video:
        video_files = [args.video]
    else:
        extensions = set(cfg["video_extensions"])
        video_files = scan_videos(args.video_dir, extensions)
    if not video_files:
        sys.exit(f"ERROR: no video files found in: {args.video_dir}")

    device = args.device or (
        "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    )

    print(f"Found {len(video_files)} video(s)")
    print(f"Output : {args.output}")
    print(f"ROI    : {len(roi_polygons)} polygon(s)")
    print_device_info(device, cfg["frame"]["frame_skip"])

    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(video_files, start=1):
        video_output_folder = args.output / video_path.stem
        print(f"\n[{idx}/{len(video_files)}] {video_path.name}")
        print(f"    Output -> {video_output_folder}")

        # Reload the model per video so ByteTrack state (track IDs, track
        # history) never leaks across videos -- a batch run stays identical
        # to running each video on its own.
        model = YOLO(str(args.model))
        model.to(device)

        ok = process_video(video_path, video_output_folder, model, cfg,
                           roi_polygons, write_video=not args.no_video)
        if ok:
            success_count += 1
        else:
            fail_count += 1

        del model
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "-" * 60)
    print(f"All done. {success_count} succeeded | {fail_count} failed")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
