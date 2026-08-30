# Supplied dataset and physical setup audit

Audit date: 2026-08-30

## Reproducible inventory

Run:

```bash
PYTHONPATH=src python3 scripts/audit_count_dataset.py data/training
```

Observed dataset fingerprint:

```text
SHA-256: 592b24daeb7ca372372067d186fb390ab5d23c94cda083f050153e0caef5fc66
```

| Property | Observed value |
|---|---:|
| Valid JPEG images | 133 |
| Exact byte-for-byte duplicates | 0 |
| Invalid JPEG files | 0 |
| Count classes | 0 through 12 |
| Images in class 2 | 13 |
| Images in every other class | 10 |
| 320×320 images | 20 |
| 800×600 images | 113 |
| First filename timestamp | 2026-08-30 07:27:29.809 |
| Last filename timestamp | 2026-08-30 07:52:13.909 |

The filename timestamps span approximately 25 minutes, not multiple independent
days or environments. Consecutive frames are usually repeated observations of
the same physical car arrangement. There are no exact duplicate files because
camera noise and exposure vary, but many images are near-duplicates in content.

## What the folders mean

The directory name is the total number of cars in the complete image:

```text
data/training/0/*.jpg  -> total count 0
data/training/1/*.jpg  -> total count 1
...
data/training/12/*.jpg -> total count 12
```

This is a weak image-level classification label. It does not identify:

- the left-lane count;
- the right-lane count;
- any vehicle bounding box or centre coordinate;
- the lane of an individual vehicle;
- the junction/stop-line coordinates; or
- physical distance from a vehicle to the junction.

A total-count classifier can be explored from these labels. The intended
left/right count and proximity output requires object-level annotations and
per-camera-position geometry in addition to this dataset.

## Image-domain findings

The images confirm the expected operating scene: a white two-lane road with a
dark central divider and short dark lane marks. They also expose conditions the
runtime must handle:

- strong green/magenta colour shifts from automatic white balance;
- large exposure and brightness changes between repeated frames;
- motion/focus blur in some captures;
- wires crossing the lower part of the image;
- two large dark traffic-light housing surfaces occluding both near-lane areas;
- cars partially hidden by the housing;
- dark lane markings that resemble narrow detected objects;
- white/silver cars with low contrast against the road; and
- cars placed close together or touching.

Classes 0 and 1 are captured at 320×320 while all later classes are 800×600.
This creates a direct label-correlated shortcut: a model can separate counts
0–1 from counts 2–12 using resolution/crop characteristics rather than cars.
All training inputs must be normalized, and future collection must use one
fixed camera resolution and field of view for every count.

## Physical setup interpretation

The supplied overview shows a four-arm tabletop crossroad rather than a true
circulating roundabout. A central traffic-light housing contains signal heads,
with an ESP32-class camera mounted above it on servo hardware. The intended
capture process rotates the camera to one road at a time.

Important deployment consequences are:

1. Each motor position has a different image-to-road transform and therefore
   needs its own left/right polygons and junction line.
2. The current local dataset represents only one apparent camera position. It
   cannot validate performance on the other three roads.
3. The central housing blocks the near portion of the lanes. Vehicles hidden
   completely by it cannot be recovered by any image model.
4. Cables in front of the lens create non-road foreground objects and should be
   routed outside the camera field before final collection.
5. The colourful printed play mats surrounding the white roadway can create a
   different background at other motor positions.
6. Servo settling, focus, and exposure should be completed before capture.
7. The camera position must be repeatable; otherwise saved lane polygons and
   perspective calibration will drift.

The production capture sequence should include the motor position/road ID in
metadata rather than asking the model to infer it.

## Pretrained detector probe

A YOLO11n COCO model was tested at 800-pixel input. On a representative
12-vehicle image it reported toy cars mostly as `bottle`, along with false
`keyboard`, `refrigerator`, and `toothbrush` detections. Cross-class boxes also
overlapped the same vehicle.

After broad region/shape filtering and class-agnostic suppression, the best
exploratory threshold produced an approximate total-count mean absolute error
of 1.07 and an exact result on 43 of 133 images. These are bootstrap diagnostics,
not accepted model metrics, because the probe was evaluated on the same images
used to tune its filtering and its boxes were not manually verified.

Pretrained boxes may reduce annotation work, but must not be treated as ground
truth automatically.

## Readiness decision

| Capability | Supported by supplied data? | Reason |
|---|---|---|
| Experimental total-count classifier | Partially | Total labels exist, but the dataset is tiny and highly repeated |
| Honest unseen-scene accuracy estimate | No | One short capture session and few independent arrangements |
| Left/right lane count detector | No | No boxes or per-lane labels |
| Vehicle coordinates | No | No object annotations |
| Junction proximity | No | No object coordinates or calibrated road plane |
| Four-road deployment validation | No | Only one apparent road/camera position is represented |

The data is useful as an initial engineering sample and annotation source. It is
not sufficient to declare the requested system deployment-ready.

The provisional classification experiment is configured in
`configs/count-classification.json`. It uses the generated
`data/classification` split and writes all artifacts below this repository's
`runs/classify` directory. The configuration is committed for reproducibility;
the generated split and model weights are not committed to normal Git.

## Provisional classifier result

The configured YOLO11n classification baseline was trained for 50 epochs. It
reached 30.8 percent top-1 accuracy on the 13-image validation split. On the
50-image held-out split, the count-specific results were:

| Metric | Result |
|---|---:|
| Exact total-count accuracy | 16% |
| Within one car | 50% |
| Mean absolute error | 1.96 cars |
| Root mean squared error | 2.67 cars |
| Mean signed error | +0.72 cars |

This baseline is rejected. Its checksum, configuration, data fingerprint, and
decision are recorded in `configs/model-evaluation.count-baseline.json`. These
metrics are not lane metrics and cannot be used to claim that the requested
left/right system works. The result supports moving to object annotations and
geometry instead of attempting to infer locations from total labels.

A second controlled experiment increased the input from 224 to 320 pixels and
disabled random crop/erase augmentation so that training always sees the full
road. It improved exact total-count accuracy to 30 percent and MAE to 1.66 cars,
but remains a research-only artifact. Its record is
`configs/model-evaluation.count-full-frame.json`; its local ignored artifact is
`artifacts/research/count-full-frame.pt`. It must not be connected to the MCDM
controller.

## Minimum data completion package

Before an accepted detector is trained, supply or create:

1. Bounding boxes around every visible car in every usable image.
2. A reviewed rule for partially occluded vehicles.
3. Empty and populated captures from all four motor positions.
4. One fixed runtime resolution, preferably the current 800×600 or higher.
5. Several independent arrangements per count and lane distribution.
6. Separate collection sessions with changed lighting and camera restarts.
7. Four calibration images with known road-plane reference points.
8. A lane-capacity value and visible physical lane length for every road view.

The repository includes validation, training, evaluation, and calibration tools
for this completion package. The next code milestones add a safe provisional
count-classification experiment while keeping it separate from the detection
pipeline.
