"""Sidewalk-intrusion detection: ROI polygon parsing and geometry tests.

A violation is flagged when the bottom strip of a tracked motorcycle's
bounding box falls inside the sidewalk ROI polygon of the camera location
(temporal confirmation is handled by inference.violation_tracker).
"""

from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_yolo_roi(yolo_str: str) -> list:
    """Parse YOLO segmentation format into normalized polygons.

    Input : "class_id x1 y1 x2 y2 ..." (one line per polygon, coords 0-1)
    Output: list of np.ndarray with shape (N, 2), dtype float32
    """
    polygons = []
    for line in yolo_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        values = list(map(float, line.split()))
        coords = values[1:]
        if len(coords) < 6:
            print("WARNING: ROI line skipped: a polygon needs at least 3 points (6 values)")
            continue
        if len(coords) % 2 != 0:
            print("WARNING: ROI line skipped: coordinate count must be even")
            continue
        points = np.array(
            [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)],
            dtype=np.float32,
        )
        polygons.append(points)
    return polygons


def scale_polygons(normalized_polygons: list, frame_size: tuple = (1920, 1080)) -> list:
    """Scale normalized polygons to pixel coordinates in OpenCV contour shape."""
    w, h = frame_size
    return [
        (poly * [w, h]).astype(np.int32).reshape((-1, 1, 2))
        for poly in normalized_polygons
    ]


def load_roi(roi_yaml_path, location: str, frame_size: tuple = (1920, 1080)) -> list:
    """Load a location's ROI from configs/roi.yaml, scaled to pixel coordinates."""
    data = yaml.safe_load(Path(roi_yaml_path).read_text(encoding="utf-8"))
    locations = data.get("locations", {})
    if location not in locations:
        raise KeyError(
            f"ROI location '{location}' not found in {roi_yaml_path} "
            f"(available: {sorted(locations)})"
        )
    return scale_polygons(parse_yolo_roi(locations[location]), frame_size)


def check_roi_multipoint(
    rx1, ry1, rx2, ry2,
    roi_polygons,
    ratio: float = 0.1,
    n_points: int = 5,
    require: int = 3,
) -> bool:
    """Sample points along the bbox bottom strip; True if >= `require` fall inside the ROI."""
    if not roi_polygons:
        return False

    r_width = rx2 - rx1
    check_y = float(ry2) - (ry2 - ry1) * ratio

    if n_points == 1:
        check_points = [(float(rx1) + r_width * 0.5, check_y)]
    else:
        check_points = [
            (float(rx1) + r_width * (i / (n_points - 1)), check_y)
            for i in range(n_points)
        ]

    count = sum(
        1 for pt in check_points
        if any(
            cv2.pointPolygonTest(poly, pt, False) >= 0
            for poly in roi_polygons
        )
    )
    return count >= require


def draw_roi_overlay(frame, roi_polygons, color: tuple = (0, 165, 255), alpha: float = 0.3):
    """Draw the ROI as a semi-transparent fill with a solid outline."""
    overlay = frame.copy()
    for poly in roi_polygons:
        cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for poly in roi_polygons:
        cv2.polylines(frame, [poly], True, color, 2)
