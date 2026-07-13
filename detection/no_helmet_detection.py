"""No-helmet detection: associate helmet-class detections with tracked riders.

A helmet/no-helmet box belongs to a rider when its center lies inside the
upper half of the rider's bounding box. A rider with at least one matched
"no helmet" box is flagged (temporal logging rules live in
inference.violation_tracker).
"""

NO_HELMET_CLASSES = {"no_helmet", "no-helmet", "nohelmet", "no helmet"}


def normalize_cls(cls_name: str) -> str:
    return cls_name.lower().replace(" ", "_").replace("-", "_")


def split_detections(boxes, class_names) -> tuple:
    """Split YOLO tracking results into rider tracks and helmet-class boxes.

    Returns (riders, helmets, skipped_no_id):
      riders  = [(xyxy, track_id)] for rider/motorcycle boxes with a track id
      helmets = [(xyxy, cls_name, conf)] for every other class
      skipped_no_id = rider/motorcycle boxes dropped for lacking a track id
    """
    riders = []
    helmets = []
    skipped_no_id = 0

    for box in boxes:
        b_xyxy = box.xyxy[0].cpu().numpy()
        c_idx = int(box.cls[0].cpu().numpy())
        cls_name = class_names[c_idx]
        conf = float(box.conf[0].cpu().numpy())

        if "rider" in cls_name.lower() or "motorcycle" in cls_name.lower():
            if box.id is not None:
                riders.append((b_xyxy, int(box.id[0].cpu().numpy())))
            else:
                skipped_no_id += 1
        else:
            helmets.append((b_xyxy, cls_name, conf))

    return riders, helmets, skipped_no_id


def match_helmets_to_rider(rider_box, helmets, matched_indices: set) -> list:
    """Return the helmet boxes whose center lies in the rider bbox's upper half.

    Matched helmet indices are added to `matched_indices` so each helmet box
    is assigned to at most one rider per frame.
    """
    rx1, ry1, rx2, ry2 = map(int, rider_box)
    mid_y = ry1 + ((ry2 - ry1) // 2)

    matched = []
    for h_idx, (h_box, h_cls, h_conf) in enumerate(helmets):
        if h_idx in matched_indices:
            continue
        hx1, hy1, hx2, hy2 = map(int, h_box)
        h_cx = (hx1 + hx2) // 2
        h_cy = (hy1 + hy2) // 2
        if (rx1 < h_cx < rx2) and (ry1 < h_cy < mid_y):
            matched.append((h_box, h_cls, h_conf))
            matched_indices.add(h_idx)
    return matched


def has_no_helmet(matched_helmets, no_helmet_classes=NO_HELMET_CLASSES) -> bool:
    """True if any helmet box matched to this rider is a 'no helmet' class."""
    return any(
        normalize_cls(h_cls) in no_helmet_classes
        for (_, h_cls, _) in matched_helmets
    )
