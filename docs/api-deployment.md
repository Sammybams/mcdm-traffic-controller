# FastAPI testing, Azure App Service, and Render deployment

## What is available

The API loads one detector when the process starts and serializes inference
through a lock. This avoids loading the model for every request and avoids
concurrent access to the same Ultralytics model instance.

| Method and path | Purpose |
|---|---|
| `GET /` | Service discovery |
| `GET /health` | Deployment readiness and model-loaded check |
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
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m uvicorn traffic_vision.api:app \
  --app-dir src \
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

The health endpoint intentionally remains unauthenticated so the hosting
platform can probe it. Leaving the API-key variable unset is convenient locally
but exposes public inference on a deployed service.

## Runtime environment

| Variable | Default | Purpose |
|---|---|---|
| `TRAFFIC_VISION_MODEL` | `artifacts/research/toy-vehicle-prelabel.pt` | Detector artifact path |
| `TRAFFIC_VISION_ROADS_CONFIG` | `configs/roads.supplied-view-provisional.json` | Four-road geometry |
| `TRAFFIC_VISION_CONFIDENCE` | `0.20` | Accepted detection threshold |
| `TRAFFIC_VISION_MAX_UPLOAD_MB` | `10` | Per-image upload limit |
| `TRAFFIC_VISION_API_KEY` | unset | Optional `X-API-Key` value |

## Runtime model artifact

The required 5.2 MB runtime model is committed at
`artifacts/research/toy-vehicle-prelabel.pt`. Generated training weights,
proposal models, datasets, and the rejected count classifier remain excluded.
The committed artifact's SHA-256 is:

The expected SHA-256 is:

```text
682b580b0bdb8afd9c42b3202a238bd31942ffa7c33f37ebb81ce2a268828544
```

GitHub Actions verifies this checksum before Azure deployment. The model
evaluation record contains the same checksum so the deployed binary remains
traceable to its measured results.

## Deploy to Azure App Service

The Azure-generated workflow at `.github/workflows/main_traffic.yml` now:

1. selects Python 3.13;
2. installs `requirements.txt`;
3. runs the complete test suite;
4. verifies the committed model checksum;
5. authenticates to the existing Azure Web App using GitHub OIDC;
6. deploys the complete repository artifact; and
7. configures one Uvicorn worker with the correct FastAPI import path.

Every push to `main` triggers the workflow. In the Azure Portal, open the
`traffic` Web App, then Settings > Environment variables, and add:

```text
TRAFFIC_VISION_API_KEY=<your generated secret>
```

The other runtime variables have safe defaults for the committed model and
provisional geometry. After the workflow succeeds, test:

```text
https://traffic.azurewebsites.net/health
https://traffic.azurewebsites.net/docs
```

If the Web App has a different public hostname, use the hostname shown on its
Azure Overview page. Send `X-API-Key` on inference requests when the setting is
enabled.

## Deploy with the Render Blueprint

The root `render.yaml` defines a Python web service, the Uvicorn start command,
the `/health` readiness check, committed model path, and environment variables.

1. In Render, create a new Blueprint and connect this GitHub repository.
2. Set `TRAFFIC_VISION_API_KEY` to a long random value and retain it for the
   camera/controller client.
3. Create the Blueprint and wait for the build and health check to pass.
4. Visit `https://YOUR-SERVICE.onrender.com/health` and `/docs`.
5. Repeat the local `curl` calls with the Render hostname and `X-API-Key`.

The Blueprint selects the `1c-2g` plan because PyTorch and Ultralytics are
unlikely to fit reliably in a 512 MB instance. One worker is intentional: every
worker would load another copy of the model. Uploaded images do not need a
persistent disk because requests use temporary storage and the model is part of
the deployment artifact.

Render deployment proves that the service starts and accepts requests. It does
not replace four-position camera calibration or independent model acceptance
testing described in `docs/remaining-implementation-plan.md`.
