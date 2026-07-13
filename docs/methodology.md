# Methodology

## 1. Problem statement & violation definitions

Two violations are detected per tracked motorcycle, with the following
priority when both could apply in the same frame (highest first):

1. **Sidewalk Intrusion (Stationary)**: always logged
2. **Sidewalk Intrusion**: always logged
3. **No Helmet Violation**: logged only while the vehicle is moving and not
   already flagged for sidewalk intrusion
4. **Stationary Vehicle**: shown in the annotated video, never logged
5. **Normal**: not logged

A vehicle is confirmed "on the sidewalk" once its bounding box has satisfied
the ROI test (§7) for `sidewalk.confirm_threshold` consecutive frames
(default 10). It's considered "stationary" once its tracked center has moved
less than `motion.position_tolerance` pixels for
`motion.stationary_threshold` consecutive frames (default 180); this
suppresses false no-helmet violations for parked motorcycles. Wrong-way
riding was measured in the underlying study but is out of scope for this
repository's detection code.

## 2. Data collection & cleaning criteria

Frames were captured from public Thai municipal CCTV web streams, at 16
distinct camera locations (see `configs/roi.yaml`), and cover both day
(`D` filename prefix) and night (`N` prefix) conditions.

Captured frames were then manually screened: any frame that was damaged,
blurred, or in which the objects could not be clearly identified was
removed, and only frames actually containing a motorcycle rider were kept
(a frame with no motorcycle rider was discarded). This was a one-time visual
judgment call on the raw captures, not a scripted or threshold-based filter,
and the rejected frames were not retained, so there is no surviving sample
of "noise" images to derive a quantitative rule from after the fact, and no
reusable `clean_and_filter.py`. **2,017 images** passed screening. Images
were not manually resized; YOLO's `imgsz` setting rescales them at load time.

## 3. Annotation & classes

Bounding boxes were drawn with LabelImg. Three classes, YOLO format:

| id | name          | box covers |
|----|---------------|------------|
| 0  | motorcycle    | the whole motorcycle **and** its rider (with any passenger) |
| 1  | with_helmet   | the helmeted head only; must lie inside a `motorcycle` box |
| 2  | no_helmet     | the bare head only; must lie inside a `motorcycle` box |

The rule that a head box must sit inside its motorcycle box is what makes the
rider↔helmet association in `detection/no_helmet_detection.py` reliable: a
helmet/no-helmet box is matched to the rider whose bounding-box upper half
contains its center.

## 4. Split strategy

**Hold-out**: the 2,017 screened images are split 80/10/10 into
train/validation/test (≈1,613 / ≈202 / ≈202). The 5-fold CV below runs on
the train+validation pool (≈1,815 images).

**5-Fold Cross-Validation**: `MultilabelStratifiedKFold` (K=5, seed=12) over
the feature vector

```
[is_day, is_night, motorcycle_count, with_helmet_count, no_helmet_count]
```

per image, so every fold preserves both the day/night ratio and the
per-class instance distribution; plain random K-Fold would risk folds with
skewed lighting or class balance given the relatively small dataset.

**Integrity rule**: every image must appear in the training split exactly
`K-1` times and in the validation split exactly once, across all folds
combined. `data_preparation/stratified_kfold_split.py --verify-only` checks
this automatically.

## 5. Training setup

All six model variants (YOLOv8n/s, YOLO11n/s, YOLO26n/s) are trained with an
identical, fixed hyperparameter set (`configs/hyperparameters.yaml`) so that
comparisons reflect architecture differences, not tuning differences:

| Hyperparameter | Value |
|---|---|
| optimizer | AdamW |
| lr0 / lrf | 0.005 / 0.1 |
| dropout | 0.2 |
| weight_decay | 0.0005 |
| imgsz | 512 |
| batch | 16 |
| epochs | 100 |
| patience | 20 |

The fixed set was chosen once, from GPU-memory-constrained experimentation
on the hardware available for this project (single consumer GPU), rather
than a full per-model hyperparameter sweep; cross-model comparability was
prioritized over squeezing the last few points of mAP out of any one
variant.

## 6. Evaluation protocol

Two distinct evaluations:

**Object detection**: how well YOLO detects the three classes.

- **K-Fold CV**: `model.val(split='val')` per fold per variant, then mean and
  sample standard deviation (`ddof=1`) across the 5 folds for each metric.
- **Hold-out test**: `model.val(split='test')` per trained model, reporting
  Precision, Recall, F1 (`2PR/(P+R)`) and mAP50 (the report uses these four;
  the run CSVs also carry mAP50-95 and inference speed).

**Violation detection (system)**: how well the end-to-end pipeline flags the
two violations, evaluated on real test-video cases (Sidewalk Intrusion: 104
cases; No Helmet: 338 cases). Scored by Precision / Recall / F1, with F1 as the
main selection criterion. YOLO26s was the best model for both violations.

## 7. Inference / ROI methodology

Sidewalk ROI polygons are defined per camera location in YOLO segmentation
format (normalized 0-1 coordinates), stored in `configs/roi.yaml`.

A tracked vehicle's bounding box is tested against the ROI with a
**multipoint bottom-strip test** (`detection/sidewalk_detection.check_roi_multipoint`):

1. Sample `roi_check_n_points` (default 5) points evenly across the bbox
   width, at height `roi_check_ratio` (default 0.1, i.e. the bottom 10%) up
   from the bottom edge.
2. Count how many sampled points fall inside any ROI polygon
   (`cv2.pointPolygonTest`).
3. The bbox is considered "in ROI" for that frame if at least
   `roi_check_require` (default 3) of the 5 points are inside.

This is more robust to partial occlusion and bbox jitter than testing a
single center point or the full bbox overlap, since only the *base* of the
vehicle (where it contacts the ground) needs to be on the sidewalk.

Tracking uses Ultralytics' ByteTrack integration (`model.track(...,
tracker="bytetrack.yaml")`), which assigns persistent IDs across frames so
that the confirmation counters in §1 accumulate per physical vehicle rather
than per detection.
