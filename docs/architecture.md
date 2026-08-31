# Runtime and model infrastructure

## Scope boundary

The repository is the sensing layer. It receives four separate photographs,
detects toy vehicles, and calculates deterministic measurements. It does not
choose or activate a traffic-light phase.

The MCDM controller should consume the result as an input and remain a separate
service or firmware component.

## Runtime flow

```text
Motor reaches configured road position
        |
Camera captures one road image
        |
Object detector returns vehicle bounding boxes
        |
Bounding-box centres are normalized to image coordinates
        |
Per-road homography maps coordinates onto the road plane
        |
Lane polygons assign each vehicle to left or right
        |
Distance to the configured junction line is calculated
        |
Per-lane count, density, nearest distance, and proximity are calculated
        |
The four independent road results are returned as one junction result
        |
Separate MCDM controller consumes the measurements
```

## Component boundaries

| Component | Responsibility | Does not do |
|---|---|---|
| Camera/motor controller | Select road, settle camera, capture and identify image | Detect vehicles or choose phases |
| `VehicleDetector` | Return bounding boxes, labels, and confidence | Understand lanes, density, or MCDM |
| Road configuration | Store lane polygons, stop line, homography, capacities | Change at inference time |
| Road processor | Assign boxes and calculate measurements | Learn from images |
| Four-road service | Require and combine four matching road IDs | Treat captures as simultaneous |
| MCDM controller | Apply the agreed weighted formula and history | Run object detection |
| Safety controller | Enforce valid signal transitions and fail-safe timing | Trust an unconstrained score directly |

## Measurement definitions

For each lane:

```text
count = number of accepted detections assigned to the lane
density = min(1, count / configured maximum lane capacity)
nearest_distance = minimum mapped distance to the junction line
proximity = clamp(1 - nearest_distance / visible_length, 0, 1)
```

An empty lane has zero count, zero density, `null` nearest distance, and zero
proximity. The road-level values use the combined capacity of both lanes and
the closest assigned vehicle from either lane.

If only the approaching lane should affect the MCDM formula, that semantic
mapping belongs in the controller configuration. The vision system deliberately
reports physical `left` and `right` lanes.

## Coordinate systems

There are three coordinate systems:

1. Detector bounding boxes use image pixels.
2. Box centres are divided by image width and height to create normalized image
   coordinates in the range zero to one.
3. A per-road 3 by 3 homography maps normalized points to the calibrated road
   plane.

The example configuration uses an identity homography and reports normalized
distance. It is not a physical calibration. For centimetres, record four known
image-to-board point pairs per camera position, compute the homography, and put
lane polygons, junction line, and visible length in centimetres.

## Model infrastructure

### Current experiment environment

The supplied-data classification and detection experiments were run locally
with Python 3.13.7, Ultralytics 8.4.135, PyTorch 2.13.0, and an Apple M4 Pro.
Ultralytics reported CPU execution even though `mps` was requested. AI-assisted
pre-labelling used Grounding DINO Tiny and YOLO-World Small as proposal models;
the deployed interim detector is the much smaller single-class YOLO11n model.
Proposal models are development dependencies and are not loaded at runtime.

Production requirements use the official
`ultralytics-opencv-headless==8.4.135` distribution. It exposes the same
`ultralytics` Python API while replacing desktop OpenCV with the headless wheel.
This is required by the Azure Linux image, which does not include the X11
`libxcb.so.1` library needed by desktop OpenCV.

This local setup is adequate for the tiny research set; final training should
use a supported GPU when the reviewed dataset grows. Exact weights,
configuration checksums, confidence threshold, data fingerprints, and results
are in the `configs/model-evaluation.*.json` records.

### Offline path

```text
Final camera rig
  -> capture sessions
  -> immutable original images
  -> AI-proposed bounding boxes
  -> independent human review/correction
  -> annotation validation
  -> session-based train/validation/test split
  -> transfer-learning experiment
  -> detector metrics plus end-to-end lane evaluation
  -> accepted best weights
  -> versioned deployment artifact
```

Training is offline and may use a GPU workstation or hosted GPU. Runtime should
be local to the junction prototype; it does not require cloud connectivity.

The model layers are deliberately separate:

| Layer | Model or algorithm | Where it runs |
|---|---|---|
| Label suggestion | Grounding DINO Tiny + YOLO-World Small | Development/training only |
| Vehicle inference | Fine-tuned one-class YOLO11n | Runtime host |
| Lane assignment | Fixed polygons and homography | Runtime host, deterministic |
| Density/proximity | Fixed arithmetic from counts and coordinates | Runtime host, deterministic |
| Traffic choice | Existing weighted MCDM formula and history | Separate controller |

### Edge runtime path

```text
Camera -> Python process -> model runtime -> geometry pipeline -> JSON result
```

The current adapter loads an Ultralytics-compatible `.pt` model. The interface
allows a later ONNX, OpenVINO, TensorRT, or another detector adapter without
changing the geometry or output contract.

The rotating camera introduces temporal skew: four road photographs are not a
single simultaneous observation. The versioned capture-manifest contract stores
the road ID, motor position, timestamp, and image path for each capture and
rejects a cycle that exceeds the configured maximum span. Servo movement,
settling, exposure, and inference should be optimized so the complete cycle is
short relative to traffic movement.

The current demonstration API runs on Azure App Service; the final tabletop can
instead run on a laptop, Raspberry Pi-class edge computer, or another local host
beside the controller. Hosting location does not change the ownership boundary:
the motor/camera firmware owns rotation and capture, the Python service owns
detection and geometry, the MCDM process owns scoring, and the safety/relay
controller owns legal red-amber-green transitions. A model result must never
directly energize conflicting green lights.

### Artifact ownership

| Asset | Versioned in normal Git? | Recommended location |
|---|---:|---|
| Source code and tests | Yes | Git repository |
| Road/training configuration | Yes | `configs/` |
| Data rules and experiment notes | Yes | `docs/` |
| Original and labelled images | No | DVC, Git LFS, or versioned object storage |
| Training runs and plots | No | Experiment/artifact storage |
| Model weights | Usually no | Model registry or release artifact storage |
| Deployment model checksum/version | Yes | Release manifest |

The current 5.2 MB provisional detector is a deliberate exception: its `.pt`
file is committed with a fixed checksum so Azure and Render deployments are
self-contained. Larger, intermediate, rejected, and proposal-model weights
remain outside normal Git history.

Every deployed result should be traceable to a source commit, dataset version,
training configuration, model checksum, road calibration version, and detector
confidence threshold.

## Failure handling

- A detection below the configured confidence threshold is ignored.
- A valid detection outside both lane polygons is counted as unassigned.
- Missing or mismatched road IDs reject the complete four-road batch.
- An empty lane produces explicit empty metrics instead of a guessed distance.
- Camera blur, motor-position validation, timestamps, and maximum data age are
  expected from the capture layer and should be added before hardware control.
- The downstream controller must use fixed-cycle or safe fallback behaviour if
  the sensing result is absent, stale, or invalid.
