"""Per-track violation state: temporal confirmation, logging, and snapshots."""

import math
import os

import cv2
import pandas as pd


class ViolationTracker:
    """
    Priority logic (high -> low):
      1. Sidewalk Intrusion (Stationary)  <- always logged
      2. Sidewalk Intrusion               <- always logged
      3. No Helmet Violation              <- logged only while moving and not on sidewalk
      4. Stationary Vehicle               <- displayed only, never logged
      5. Normal                           <- not logged
    """

    def __init__(
        self,
        output_folder: str,
        sidewalk_threshold: int = 10,
        stationary_threshold: int = 180,
        tolerance: int = 3,
    ):
        self.tracker_db = {}
        self.violation_log = []

        self.output_folder = output_folder
        self.sidewalk_threshold = sidewalk_threshold
        self.stationary_threshold = stationary_threshold
        self.tolerance = tolerance

    def update_vehicle(self, track_id, bbox, is_in_roi, tracking_center):
        if track_id not in self.tracker_db:
            self.tracker_db[track_id] = {
                "sidewalk_count": 0,
                "stationary_count": 0,
                "last_center": tracking_center,
                "is_in_roi": False,
                "logged_helmet": False,
                "logged_sidewalk": False,
                "violation_type": "Normal",
                "bbox": bbox,
            }

        record = self.tracker_db[track_id]
        record["bbox"] = bbox
        record["is_in_roi"] = is_in_roi

        if is_in_roi:
            record["sidewalk_count"] += 1
        else:
            record["sidewalk_count"] = 0

        last_c = record["last_center"]
        dist = math.hypot(
            tracking_center[0] - last_c[0],
            tracking_center[1] - last_c[1],
        )
        record["last_center"] = tracking_center

        if dist < self.tolerance:
            record["stationary_count"] += 1
        else:
            # a vehicle that starts moving again may commit a fresh helmet violation
            was_stationary = record["stationary_count"] >= self.stationary_threshold
            if was_stationary:
                record["logged_helmet"] = False
            record["stationary_count"] = 0

    def cleanup_old_tracks(self, active_ids: set):
        stale_ids = set(self.tracker_db.keys()) - active_ids
        for sid in stale_ids:
            del self.tracker_db[sid]

    def check_and_log(self, track_id, frame, video_time_str,
                      video_time_file, is_any_no_helmet, frame_count) -> str:
        record = self.tracker_db[track_id]

        is_sidewalk_confirmed = record["sidewalk_count"] >= self.sidewalk_threshold
        is_stationary = record["stationary_count"] >= self.stationary_threshold

        if is_sidewalk_confirmed:
            current_status = (
                "Sidewalk Intrusion (Stationary)" if is_stationary
                else "Sidewalk Intrusion"
            )
        elif is_any_no_helmet and not is_stationary:
            current_status = "No Helmet Violation"
        elif is_stationary:
            current_status = "Stationary Vehicle"
        else:
            current_status = "Normal"

        record["violation_type"] = current_status

        if is_sidewalk_confirmed and not record["logged_sidewalk"]:
            record["logged_sidewalk"] = True
            self._log_and_snapshot(track_id, current_status,
                                   video_time_str, frame,
                                   video_time_file, frame_count, record)

        # independent `if` (not elif): a sidewalk violator can also ride helmetless
        if is_any_no_helmet and not is_stationary and not record["logged_helmet"]:
            record["logged_helmet"] = True
            if not is_sidewalk_confirmed:
                self._log_and_snapshot(track_id, "No Helmet Violation",
                                       video_time_str, frame,
                                       video_time_file, frame_count, record)

        return current_status

    def _log_and_snapshot(self, track_id, violation_type, video_time_str,
                          frame, video_time_file, frame_count, record):
        self.violation_log.append({
            "Timestamp": video_time_str,
            "Vehicle_ID": int(track_id),
            "Violation": violation_type,
        })
        self._save_snapshot(track_id, frame, video_time_file, frame_count, record)

    def _save_snapshot(self, track_id, frame, video_time_file, frame_count, record):
        x1, y1, x2, y2 = map(int, record["bbox"])
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)

        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size > 0:
            snap_dir = os.path.join(self.output_folder, "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            snap_name = os.path.join(
                snap_dir,
                f"ID_{int(track_id)}_{video_time_file}_f{frame_count}.jpg",
            )
            cv2.imwrite(snap_name, crop_img)

    def export_report(self, filepath: str):
        if self.violation_log:
            df = pd.DataFrame(self.violation_log)
            df.to_csv(filepath, index=False)
            print(f"    Report saved -> {filepath}")
        else:
            print("    No violations found.")
