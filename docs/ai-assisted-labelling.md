# AI-assisted vehicle labelling

## What was automated

The supplied dataset did not contain object-detection labels. Its directory
names contained only the total number of cars visible in each image:

```text
data/training/0/*.jpg   -> zero cars in the complete image
data/training/1/*.jpg   -> one car in the complete image
...
data/training/12/*.jpg  -> twelve cars in the complete image
```

The labelling workflow converted those weak total-count labels into provisional
YOLO bounding-box labels. It used two zero-shot vision models to suggest boxes,
deterministic rules to filter and rank the suggestions, the known total count to
limit how many boxes were selected, and explicit reviewed corrections for two
difficult arrangements.

This was **AI-assisted pre-labelling**, not fully automatic creation of trusted
ground truth. The known total can confirm that a file contains the expected
number of boxes, but it cannot confirm that every box surrounds the correct car.

The workflow never generated left/right lane labels. It generated vehicle boxes
only. At runtime, fixed lane polygons and calibrated geometry assign each
detected box centre to the left or right lane.

## Actual data lineage

```text
133 JPEGs in numeric count folders
  -> dataset audit and SHA-256 fingerprint
  -> Grounding DINO proposals
  -> YOLO-World proposals
  -> normalized scene/shape filtering
  -> cross-model agreement scoring
  -> overlap suppression
  -> select at most the known folder count
  -> one YOLO label file and review overlay per image
  -> temporal-group repair check (zero repairs were required)
  -> reviewed overrides for 10 difficult frames
  -> representative contact-sheet review
  -> 70 train / 13 validation / 50 test images
  -> YOLO11n fine-tuning
  -> validation-only confidence selection
  -> held-out total-count evaluation
  -> promoted provisional detector
```

Traceability values for the executed experiment are:

| Item | Recorded value |
|---|---|
| Source images | 133 |
| Source dataset SHA-256 | `592b24daeb7ca372372067d186fb390ab5d23c94cda083f050153e0caef5fc66` |
| Geometry-filtered candidates reported | 2,051 |
| Final provisional boxes | 786 |
| Originally selected boxes with cross-model agreement | 691 |
| Empty images and empty label files | 10 |
| Temporally propagated repairs | 0 |
| Reviewed override images | 10 |
| Temporal arrangement groups | 23 |
| Train / validation / test images | 70 / 13 / 50 |
| Final label-tree SHA-256 | `dd17d779c8e85b267c5635bc71bdbe3619708609733aad1cbd6711483387064c` |
| Runtime model SHA-256 | `682b580b0bdb8afd9c42b3202a238bd31942ffa7c33f37ebb81ce2a268828544` |

The committed experiment record is
`configs/model-evaluation.detector-prelabel.json`. Generated images, labels,
overlays, downloaded proposal weights, and training runs remain local and are
ignored by Git.

## Step 1: audit the weakly labelled source data

`scripts/audit_count_dataset.py` checks that the numeric folders and JPEG files
are readable, counts the classes and resolutions, detects byte-identical files,
extracts capture times from filenames, and calculates a deterministic dataset
fingerprint.

This established that there were 133 valid images covering counts 0 through 12,
captured during one short session. The audit did not create boxes.

## Step 2: ask two models for candidate boxes

`scripts/generate_ai_prelabels.py` constructs both proposal providers for every
non-empty image.

### Grounding DINO Tiny

`IDEA-Research/grounding-dino-tiny` receives the text prompt `toy car`.
The adapter uses:

- box threshold `0.03`;
- text threshold `0.03`; and
- non-maximum suppression IoU `0.30`.

The deliberately low thresholds favour recall. False candidates can be removed
later, but a car that neither proposal model sees cannot be recovered by the
ranking stage.

### YOLO-World Small

`yolov8s-worldv2.pt` receives five prompts:

```text
toy car
small toy car
miniature car
model car
toy vehicle
```

The adapter uses 800-pixel inference, confidence `0.003`, class-agnostic NMS at
IoU `0.30`, and a maximum of 100 candidates. Its low confidence threshold is
also intentional because this stage proposes review candidates rather than
making a production decision.

Images from folder `0` bypass both proposal models and receive an empty YOLO
label file.

## Step 3: filter and rank proposals deterministically

The two proposal lists are combined. `traffic_vision.prelabel` then applies a
fixed normalized heuristic fitted to the supplied camera view. A candidate is
retained only when all these conditions hold:

| Property | Accepted range |
|---|---:|
| Box width / image width | `0.025` to `0.18` |
| Box height / image height | `0.04` to `0.30` |
| Width-to-height aspect ratio | `0.32` to `1.60` |
| Box area / image area | `0.0015` to `0.035` |
| Normalized centre X | `0.06` to `0.82` |
| Normalized centre Y | `0.08` to `0.68` |

These bounds are a scene-specific region-and-shape heuristic. They are not the
four calibrated runtime lane polygons.

For every remaining proposal:

1. A proposal from the other model counts as agreement when their IoU is at
   least `0.45`.
2. The ranking score is the proposal confidence plus `0.35` when cross-model
   agreement exists.
3. Candidates are sorted from highest to lowest score.
4. A candidate is skipped when it overlaps an already selected box by IoU
   `0.30` or more.
5. Selection stops when the number of boxes equals the total encoded by the
   source folder.

If fewer valid candidates remain than the expected total, the image is marked
incomplete. If the expected number is reached, it is marked count-complete but
still `review_required=true`.

Each selected box is written in one-class YOLO format:

```text
0 x_center y_center width height
```

All coordinates are normalized to zero through one. An overlay is also written:

- green means the selected proposal had cross-model agreement;
- orange means it came from only one proposal source; and
- the header shows expected count, selected count, and review status.

## Step 4: check safe temporal propagation

Consecutive files were grouped by count and filename timestamp using a 30-second
gap. `scripts/repair_ai_prelabels.py` can repair an incomplete frame only when a
count-complete frame exists in the same temporal group. It copies labels from
the nearest such frame and produces a cyan review overlay.

This assumes that the arrangement did not change within that burst, so every
propagated result still requires visual review. In the executed `prelabels-v2`
run, this stage reported **zero repairs**; no final label came from automatic
temporal propagation.

## Step 5: apply reviewed corrections

Review exposed two arrangements on which the zero-shot suggestions were wrong:

- count 3, temporal group 1: both models missed two dark cars in the right lane;
- count 4, temporal group 0: proposals included background regions and missed
  dark right-lane cars.

Normalized boxes were reviewed and committed in
`configs/prelabel-overrides.json`. Each arrangement contained five repeated
frames, so `scripts/apply_prelabel_overrides.py` replaced the generated labels
for 10 images. Override overlays are purple. The script verifies that each
override contains exactly the count encoded by that group.

These corrections improve the provisional set, but they are not an independent
second-person audit of all 786 boxes.

## Step 6: review representative overlays

`scripts/make_prelabel_contact_sheet.py` chooses one overlay from each temporal
arrangement group and builds a contact sheet. The supplied session produced 23
groups. This made systematic errors easier to see than opening 133 files one by
one, but the representative sheet does not prove every repeated frame is
correct.

## Step 7: materialize the detector dataset

`scripts/prepare_detection_dataset.py` copies each source image and matching
label into YOLO `train`, `val`, or `test` directories and writes `dataset.yaml`
and `split-manifest.json`.

Captures separated by more than 30 seconds form different temporal groups. The
split planner tries to keep groups together. However, the supplied data has too
few independent arrangements: all count classes 0 through 12 are flagged in
`classes_with_group_leakage`. Therefore the 50-image test result is only a
provisional same-session result, not an honest estimate for new arrangements or
the other three camera positions.

## Step 8: train and evaluate the provisional detector

The training configuration in `configs/detection-prelabel-baseline.json` uses:

- pretrained `yolo11n.pt` detection weights;
- one class, `toy_vehicle`;
- 50 epochs;
- 640-pixel images;
- batch size 8;
- seed 42; and
- light scale, translation, and mosaic augmentation.

Confidence was swept on the 13-image validation split. The selection rule was:
lowest count MAE, then lowest empty-image false-positive rate, then highest exact
accuracy. This selected `0.20`. That fixed value was then used once on the
50-image held-out split.

The held-out total-count results were 76% exact, 98% within one car, 0.26-car
MAE, and 0% empty-image false positives. Detection precision, recall, and mAP
were also recorded, but those values compare the model with AI-assisted labels
and are not independent ground-truth measurements.

## Reproduce the pipeline

Create a new output version instead of overwriting the existing ignored
`prelabels-v2` directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,vision,autolabel]'

.venv/bin/python scripts/generate_ai_prelabels.py \
  data/training data/prelabels-v3

.venv/bin/python scripts/repair_ai_prelabels.py \
  data/training data/prelabels-v3

.venv/bin/python scripts/apply_prelabel_overrides.py \
  data/training data/prelabels-v3 configs/prelabel-overrides.json

.venv/bin/python scripts/make_prelabel_contact_sheet.py \
  data/training \
  data/prelabels-v3 \
  data/prelabels-v3/representative-contact-sheet.jpg

.venv/bin/python scripts/prepare_detection_dataset.py \
  data/training data/prelabels-v3/labels data/detection-prelabel-v3
```

The first run downloads the proposal models unless they are already cached.
Pass `--offline` to `generate_ai_prelabels.py` only when the Grounding DINO files
are already present locally; the YOLO-World weight path must also be available.

Training and threshold evaluation then use:

```bash
.venv/bin/python scripts/train_model.py \
  configs/detection-prelabel-baseline.json

.venv/bin/python scripts/evaluate_detector_counts.py \
  --model artifacts/research/toy-vehicle-prelabel.pt \
  --images data/detection-prelabel/images/val \
  --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50

.venv/bin/python scripts/evaluate_detector_counts.py \
  --model artifacts/research/toy-vehicle-prelabel.pt \
  --images data/detection-prelabel/images/test \
  --thresholds 0.20
```

To reproduce a new version exactly, update the training configuration's
`dataset_yaml` and output name instead of silently reusing the existing run.

## What still requires people and new data

The automated work reduced the amount of box drawing, but it did not remove the
need for annotation. Before this becomes an accepted controller detector:

1. A person must inspect or correct every provisional box, ideally with an
   independent second reviewer.
2. New images must cover all four fixed servo positions, independent vehicle
   arrangements, lighting changes, blur, occlusion, and empty lanes.
3. Splits must be by session and arrangement so near-identical frames never
   cross train, validation, and test.
4. A separate acceptance set needs true left/right counts and measured
   car-to-junction distances.
5. Each road position needs calibrated lane polygons, junction line,
   homography, capacity, and visible physical length.

Only after those checks can model localization, lane count, density, and
proximity be evaluated as one complete system.
