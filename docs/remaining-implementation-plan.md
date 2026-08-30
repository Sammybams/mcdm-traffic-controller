# Remaining implementation plan

## Honest completion boundary

The software path, data audit, AI-assisted box generation, provisional detector,
four-road capture contract, deterministic lane geometry, evaluation tools, and
model adapters are implemented. The detector is usable for a supervised demo.

The system is not yet deployment-ready because the provisional boxes are not
independently reviewed ground truth, the images come from one short repeated
session and one apparent motor position, and there are no physical road-plane
calibration measurements. Those are now the limiting inputs—not another model
architecture or more epochs on the same frames.

Current status:

| Area | Status |
|---|---|
| AI-assisted labels for supplied 133 images | Complete provisionally |
| YOLO11n localization/count baseline | Complete provisionally |
| Four-image left/right JSON pipeline | Working with estimated one-view geometry |
| Independent box/lane ground truth | Required |
| Distinct captures for roads 1–4 | Required |
| Per-position physical calibration | Required |
| Hardware fail-safe integration | Required |

## Smallest useful delivery increments

Treat every numbered item below as one reviewable commit. Generated images and
weights remain outside normal Git; commit their manifest/checksum and result.

### A. Complete the object-detection data

1. Commit the final partial-visibility rule and annotation-tool project settings.
2. Annotate and independently review the 20 images for counts 0 and 1.
3. Annotate and review count 2, including every partially occluded car.
4. Annotate and review counts 3 and 4.
5. Annotate and review counts 5 and 6.
6. Annotate and review counts 7 and 8.
7. Annotate and review counts 9 and 10.
8. Annotate and review counts 11 and 12.
9. Run `verify_annotation_batch.py`; commit the clean report and annotation
   export checksum.
10. Add session/arrangement IDs and commit the leakage-aware detector split
    manifest.

The existing 133 images can bootstrap detector development, but they are not an
acceptance dataset because repeated frames are not independent examples.

### B. Establish a detector baseline — provisional pass complete

11. **Done provisionally:** materialize leakage-limited YOLO
    train/validation/test directories from the AI-assisted labels.
12. **Done provisionally:** commit the first detector configuration and train
    YOLO11n at 640 pixels.
13. **Done provisionally:** commit precision, recall, mAP, count error,
    empty-road false positives, checksum, and the research-only decision.
14. Commit a 960-pixel experiment only if small-car recall is the limiting error.
15. Commit a failure-category report for white cars, black cars, touching cars,
    occlusion, blur, and lane-line false positives.

Do not tune against the final test set. Add new training examples for real
failure modes instead of repeatedly changing thresholds against the same 133
images.

### C. Capture all physical views

16. Mechanically lock the camera height and route cables outside the view.
17. Commit road 1 motor position, fixed exposure/focus settings, and reference
    capture checksum.
18. Repeat as separate commits for roads 2, 3, and 4.
19. Capture independent arrangements at all four positions in at least three
    sessions; commit session manifests, not image binaries.
20. Label a held-out four-road acceptance set and commit its checksum.

Completely hidden cars are unobservable. Move the camera higher or reduce the
central housing's occlusion before collecting the acceptance set if cars can
disappear behind it.

### D. Calibrate lane and proximity geometry

21. Measure four or more board coordinates visible in road 1 and commit its
    homography, left/right polygons, stop line, visible length, and lane capacity.
22. Repeat as separate commits for roads 2, 3, and 4.
23. Commit an overlay review showing test detections, lane assignment, and the
    stop line for each view.
24. Measure known car-to-junction distances and commit the distance-error report.

Density is `count / visible lane capacity`. Proximity is
`1 - nearest_distance / visible_length`, clamped to zero through one. Both need
the physical values above; they should not be learned by the model.

### E. Release and hardware integration

25. Sweep detector confidence on validation data and commit the selected value.
26. Run the end-to-end acceptance set and commit lane exact-count, road
    exact-count, MAE, distance error, and empty-lane false-positive metrics.
27. Commit an accepted model manifest only when every agreed gate passes.
28. Export and benchmark the accepted artifact on the actual edge computer.
29. Connect the camera/servo producer to `capture-manifest` version 1 and commit
    one replayable four-image integration fixture.
30. Connect the JSON measurements to the separate MCDM module and commit a dry-
    run test with signal outputs disabled.
31. Add stale-image, camera-failure, inference-failure, and invalid-phase tests.
32. Enable real light control only after the fail-safe controller enforces an
    all-red transition and prevents conflicting greens.

## Definition of done

The development is complete when an unseen, four-position acceptance set meets
the agreed lane-count and distance gates; the accepted model, calibration, and
source are checksum-linked; the full rotating-camera cycle meets its maximum
age; and every missing/stale/low-quality result causes a safe fixed-cycle or
all-red fallback rather than an MCDM decision.
