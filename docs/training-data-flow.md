# Training and data process

## Current data and model status

The supplied local dataset contains 133 images labelled only by total vehicle
count from 0 through 12. The source images, generated splits, run outputs, and
weights are intentionally ignored by Git, while their checksums, configuration,
and measured results are committed.

Two total-count classifiers were trained and evaluated. The better full-frame
experiment achieved 30 percent exact accuracy and 1.66-car MAE on the held-out
split. It is a research artifact, not an accepted deployment model. It cannot
produce left/right counts, vehicle coordinates, or proximity. See
`docs/supplied-dataset-audit.md` for the evidence and experiment records.

The 133 source images have been materialized locally as an annotation batch,
but no human-reviewed bounding boxes have been supplied yet. Therefore the
required object detector has not been trained and no production accuracy claim
can be made.

The implemented code provides:

- a model-independent measurement pipeline;
- a one-class YOLO annotation validator;
- a reproducible training entry point;
- an Ultralytics detector adapter; and
- end-to-end evaluation for lane counts and nearest-junction distance.

The first accepted detector must be finalized only after the camera, height,
lens, road board, lane markings, motor positions, and lighting are fixed.

## Process actually executed on the supplied data

```text
data/training/{0..12}/*.jpg
  -> structural audit and dataset fingerprint
  -> temporal grouping of repeated captures
  -> 70 train / 13 validation / 50 held-out images
  -> YOLO11n classification transfer learning
  -> count-specific held-out evaluation
  -> rejected/research-only model records
  -> flat 133-image annotation batch for the detector workflow
```

The classifier path is an experiment made possible by the image-level labels.
It is not a substitute for the production detector path below.

```text
data/annotation/images
  -> human draws one box per visible toy car
  -> exported labels checked against each known total
  -> split by arrangement/session, not adjacent frame
  -> YOLO toy-vehicle detector transfer learning
  -> per-box precision/recall and mAP
  -> four calibrated road views
  -> end-to-end left/right count and distance evaluation
  -> release gate and versioned deployment artifact
```

Prepare the existing images for CVAT, Label Studio, Roboflow, or another box
annotation tool:

```bash
python3 scripts/prepare_annotation_batch.py data/training data/annotation
```

After exporting one-class YOLO labels to `data/annotation/labels`, check both
format and expected box totals:

```bash
python3 scripts/verify_annotation_batch.py \
  data/annotation/annotation-manifest.json \
  data/annotation/labels
```

The verifier exits non-zero for a missing label or when the number of boxes
does not equal the source image's known total. This catches omissions but still
does not replace visual review of box placement.

## What the dataset must contain

The training class is:

```text
class 0: toy_vehicle
```

The dataset is not simply photographs of the supplied cars. It consists of
complete road images captured from the actual installed camera at the four
motor positions. The background, scale, perspective, blur, shadows, and lane
markings are part of the model's operating domain.

Capture independent arrangements covering:

- all four road positions;
- zero cars and every count up to lane capacity;
- balanced and unbalanced left/right counts;
- every supplied vehicle colour and body style;
- white and black vehicles against the chosen road surface;
- touching and partially overlapping vehicles;
- vehicles close to the lane divider and road boundaries;
- vehicles close to and far from the junction;
- expected bright, dim, shadowed, and reflected lighting;
- small motor-position variations;
- a limited number of blurred frames for quality/failure testing; and
- realistic distractors that may appear during demonstrations.

Repeated video frames of an unchanged arrangement do not add meaningful
diversity and must not be spread across training and test sets.

## Capture sessions

Each collection run receives an immutable session ID:

```text
session-YYYYMMDD-NN
```

Suggested image names are:

```text
session-20260810-01_road-1_arrangement-0042_frame-01.jpg
```

Maintain a session manifest containing:

```text
session_id
captured_at
camera_serial
camera_height
camera_settings
motor_position
road_id
lighting_condition
arrangement_id
operator
notes
```

Do not silently modify an original capture. Derived images and corrected labels
receive a new dataset version.

## Annotation policy

Draw one tight axis-aligned bounding box around every countable toy vehicle.

Before annotation begins, freeze these rules:

1. A vehicle is labelled when at least 50 percent is visible. This threshold can
   be changed, but it must be consistent.
2. Touching vehicles receive separate boxes.
3. Vehicles outside both usable lane regions are still labelled; geometry, not
   the detector, decides whether they count.
4. Hands, containers, road marks, lights, and shadows are never labelled as
   vehicles.
5. Empty images have a matching empty `.txt` label file.
6. Only class `0` is valid for the first model.

YOLO label rows use normalized coordinates:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.428125 0.533333 0.081250 0.144444
```

At least ten percent of labels should be independently reviewed. Review all
empty images, crowded images, boundary cases, and any image on which the model
later fails.

## Dataset layout

```text
data/
├── dataset-card.md
├── sessions.csv
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Images and labels are ignored by normal Git. Store them in DVC, Git LFS, or an
access-controlled object store. The dataset card and version/checksum should be
committed.

## Split policy

Split by capture session and arrangement ID, never by randomly selecting
adjacent frames.

A starting split is:

- 70 percent of sessions for training;
- 15 percent for validation; and
- 15 percent for the final test set.

The test set remains untouched while choosing models or confidence thresholds.
Every split should contain all four road positions and representative counts,
vehicle colours, and lighting conditions.

The useful dataset size is validation-driven. A reasonable controlled-scene
starting point is 500 to 1,000 independent training images, 150 to 250
validation images, and at least 200 test images. Collect more examples from
failure categories rather than adding near-duplicates.

## Validation before training

Install the package and optional model dependencies:

```bash
python3 -m pip install -e '.[dev,vision]'
```

Validate all labels:

```bash
python3 scripts/validate_dataset.py data/labels
```

The validator rejects unexpected classes, malformed rows, non-normalized
coordinates, and boxes that extend beyond an image boundary.

Human review is still required; format validation cannot tell whether a box is
on the correct object.

## Training flow

1. Copy `configs/dataset.example.yaml` to a dataset-version-specific file.
2. Copy `configs/training.example.json` to an experiment-specific file.
3. Record the dataset version and source commit in the experiment notes.
4. Fine-tune from pretrained detection weights; do not train from random weights
   for the first experiment.
5. Retain training plots, metrics, configuration, logs, and best/last weights in
   artifact storage.

Run training with:

```bash
python3 scripts/train_model.py configs/training.example.json
```

The example begins with a small pretrained detector, 960-pixel input, a fixed
random seed, and 100 epochs. These are experiment defaults, not guaranteed
optimal values. Benchmark at least:

- the small model at 640-pixel input;
- the small model at 960 or 1280 pixels;
- a larger model only if small-model recall is inadequate; and
- sliced inference only if vehicles still occupy too few pixels.

Improving the crop, lens, focus, mounting height, and lighting is preferable to
compensating for poor images with a much larger model.

## Evaluation flow

Detector mAP, precision, and recall are useful diagnostics, but the product
acceptance metrics are lane counts and distances.

Create an evaluation manifest from
`configs/evaluation-manifest.example.json`, then run:

```bash
python3 scripts/evaluate_model.py \
  --manifest configs/evaluation-manifest.test.json \
  --roads configs/roads.calibrated.json \
  --model artifacts/toy-vehicle-v1.pt
```

The evaluator reports:

- exact left/right lane count rate;
- exact whole-road count rate;
- count mean absolute error; and
- nearest-distance mean absolute error when physical ground truth exists.

Proposed acceptance gates for the tabletop system are:

- at least 95 percent exact lane-count rate;
- at least 90 percent exact two-lane road-count rate;
- fewer than 1 percent false-positive empty lanes;
- at least 99 percent correct lane assignment for correctly detected cars; and
- a physical-distance tolerance agreed from the scale of the board.

Report metrics by road position, count, vehicle colour, lighting, and crowded
versus uncrowded scenes. An overall average can hide a serious failure on black
or white cars.

## Confidence-threshold selection

Do not choose the confidence threshold from the final test set. Sweep candidate
thresholds on validation data and choose the value that minimizes lane-count
error while keeping empty-road false positives within the accepted limit.

Commit the selected threshold with the road and deployment configuration.

## Model release record

Each accepted model release should record:

```text
model_name
model_version
base_model_and_version
source_commit
dataset_version_and_checksum
training_config_checksum
road_calibration_version
confidence_threshold
exact_lane_count_rate
count_mae
nearest_distance_mae
weight_file_checksum
created_at
approved_by
```

Model weights should be published as a release artifact or placed in a model
registry, not committed directly to Git.

## Deployment

For the first prototype, run the `.pt` model locally through the provided
adapter. Invoke all four images as:

```bash
traffic-vision \
  --config configs/roads.calibrated.json \
  --model artifacts/toy-vehicle-v1.pt \
  --image road_1=captures/road-1.jpg \
  --image road_2=captures/road-2.jpg \
  --image road_3=captures/road-3.jpg \
  --image road_4=captures/road-4.jpg
```

For edge optimization, add another `VehicleDetector` implementation and export
the accepted weights to the runtime supported by the selected hardware. Compare
optimized and original model outputs on the same test set before deployment.

## Retraining loop

Retain images that produce:

- missed vehicles;
- duplicate detections;
- false positives;
- unassigned detections;
- incorrect lane counts; or
- unusually large distance error.

Review and label these images, create a new immutable dataset version, retrain,
and rerun the complete untouched regression set. Never overwrite the previously
accepted model; promote a new version only when it passes all gates.

## Required infrastructure

| Stage | Minimum infrastructure |
|---|---|
| Capture | Final camera, motor mount, stable lighting, road board, session manifest |
| Annotation | Bounding-box annotation tool and label-review process |
| Data storage | Versioned local/external dataset storage with backup and checksums |
| Development | Python 3.11+, Git, test environment |
| Training | GPU workstation or hosted GPU, experiment storage |
| Evaluation | Held-out images, lane truth, measured distance truth, calibrated roads |
| Runtime | Laptop or edge SBC, camera access, model artifact, local logging |
| Release | Model/artifact registry and version manifest |

The runtime does not need internet access. Keeping inference local reduces
latency and makes the tabletop demonstration independent of network quality.
