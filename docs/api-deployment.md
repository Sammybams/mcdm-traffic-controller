# FastAPI testing and Render deployment

## What is available

The API loads one detector when the process starts and serializes inference
through a lock. This avoids loading the model for every request and avoids
concurrent access to the same Ultralytics model instance.

| Method and path | Purpose |
|---|---|
| `GET /` | Service discovery |
| `GET /health` | Render readiness and model-loaded check |
| `GET /docs` | Interactive OpenAPI interface |
| `POST /v1/roads/{road_id}/measure` | Process one camera image |
| `POST /v1/measure` | Process four road images as one junction observation |

The inference endpoints accept JPEG or PNG multipart uploads. Each upload is
limited to 10 MB by default and is deleted immediately after inference.

## Test locally

The trained artifact must exist at
`artifacts/research/toy-vehicle-prelabel.pt`. Install and start the API:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[api,vision,dev]'
.venv/bin/uvicorn traffic_vision.api:app \
  --host 127.0.0.1 \
  --port 8000
```

Open `http://127.0.0.1:8000/docs` to upload images interactively, or test the
health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Process one image as the camera rotates to a road:

```bash
curl -X POST \
  -F image=@data/training/3/esp32cam_20260830_073856_170.jpg \
  http://127.0.0.1:8000/v1/roads/road_1/measure
```

Process a completed four-image capture cycle:

```bash
curl -X POST \
  -F road_1=@captures/road-1.jpg \
  -F road_2=@captures/road-2.jpg \
  -F road_3=@captures/road-3.jpg \
  -F road_4=@captures/road-4.jpg \
  http://127.0.0.1:8000/v1/measure
```

When `TRAFFIC_VISION_API_KEY` is set, include the key on either inference
request:

```bash
-H "X-API-Key: $TRAFFIC_VISION_API_KEY"
```

The health endpoint intentionally remains unauthenticated so Render can probe
it. Leaving the API-key variable unset is convenient locally but exposes public
inference on a deployed service.

## Runtime environment

| Variable | Default | Purpose |
|---|---|---|
| `TRAFFIC_VISION_MODEL` | `artifacts/research/toy-vehicle-prelabel.pt` | Detector artifact path |
| `TRAFFIC_VISION_ROADS_CONFIG` | `configs/roads.supplied-view-provisional.json` | Four-road geometry |
| `TRAFFIC_VISION_CONFIDENCE` | `0.20` | Accepted detection threshold |
| `TRAFFIC_VISION_MAX_UPLOAD_MB` | `10` | Per-image upload limit |
| `TRAFFIC_VISION_API_KEY` | unset | Optional `X-API-Key` value |

## Put the model somewhere Render can download

The model is intentionally ignored by Git. Before the first Render deploy,
upload `artifacts/research/toy-vehicle-prelabel.pt` to a GitHub Release or an
object-storage bucket and obtain a direct HTTPS download URL.

The expected SHA-256 is:

```text
682b580b0bdb8afd9c42b3202a238bd31942ffa7c33f37ebb81ce2a268828544
```

The Render build uses `scripts/download_model.py` to download to a temporary
file, verify this checksum, and move the verified artifact into place. A wrong
file or incomplete download fails the build instead of starting with unknown
weights.

## Deploy with the Render Blueprint

The root `render.yaml` defines a Python web service, the Uvicorn start command,
the `/health` readiness check, model download, and environment variables.

1. In Render, create a new Blueprint and connect this GitHub repository.
2. When prompted, set `TRAFFIC_VISION_MODEL_URL` to the direct model URL.
3. Set `TRAFFIC_VISION_API_KEY` to a long random value and retain it for the
   camera/controller client.
4. Create the Blueprint and wait for the build and health check to pass.
5. Visit `https://YOUR-SERVICE.onrender.com/health` and `/docs`.
6. Repeat the local `curl` calls with the Render hostname and `X-API-Key`.

The Blueprint selects the `1c-2g` plan because PyTorch and Ultralytics are
unlikely to fit reliably in a 512 MB instance. One worker is intentional: every
worker would load another copy of the model. Uploaded images do not need a
persistent disk because requests use temporary storage; the model is restored
by every build.

Render deployment proves that the service starts and accepts requests. It does
not replace four-position camera calibration or independent model acceptance
testing described in `docs/remaining-implementation-plan.md`.
