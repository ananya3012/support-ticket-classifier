# Deployment Guide

This document describes the deployment architecture for the Support
Ticket Classification Platform:

-   **Frontend:** static HTML, CSS and JavaScript hosted on Vercel.
-   **Backend:** FastAPI REST API intended to be hosted on Render.
-   **Communication:** the browser frontend sends HTTP requests to the
    backend API.

The frontend has been deployed to Vercel. Complete the Render backend
deployment and end-to-end checks before describing the full application
as production-deployed.

------------------------------------------------------------------------

## Table of Contents

1.  [Architecture](#architecture)
2.  [Prerequisites](#prerequisites)
3.  [Frontend Deployment on Vercel](#frontend-deployment-on-vercel)
4.  [Backend Deployment on Render](#backend-deployment-on-render)
5.  [Configure Frontend-to-Backend
    Communication](#configure-frontend-to-backend-communication)
6.  [Verify the Deployment](#verify-the-deployment)
7.  [Troubleshooting](#troubleshooting)
8.  [Local Development](#local-development)

------------------------------------------------------------------------

## Architecture

``` text
Browser
  |
  | Loads static website
  v
Vercel
  |
  | HTTPS requests (JSON)
  v
Render Web Service
  |
  v
FastAPI application (Uvicorn)
  |
  v
Ticket classification model
(TF-IDF + Logistic Regression)
```

The frontend and backend are deployed independently. Deploying the
static frontend does not deploy the Python API.

------------------------------------------------------------------------

## Prerequisites

-   A GitHub account with access to this repository.
-   A Vercel account connected to GitHub.
-   A Render account connected to GitHub.
-   Python 3.11 or a compatible version for local development.
-   A valid `requirements.txt`.
-   A working model training pipeline at `scripts/train.py`.
-   The FastAPI application entry point: `src.api.main:app`.

Before deploying, verify that the training script creates all model
artifacts required by the API and that the API can load those artifacts
after training.

------------------------------------------------------------------------

## Frontend Deployment on Vercel

The frontend files are located in the `static/` directory:

``` text
static/
├── index.html
├── script.js
└── styles.css
```

### Create the Vercel project

1.  Sign in to [Vercel](https://vercel.com/).
2.  Choose **Add New → Project**.
3.  Import the `support-ticket-classifier` GitHub repository.
4.  Configure the project as follows:

  Setting            Value
  ------------------ ------------------------------------
  Framework Preset   Other
  Root Directory     `static`
  Build Command      Leave empty for plain static files
  Output Directory   Leave empty / use the default

5.  Deploy the project.

Vercel can create production deployments when changes are pushed to the
configured production branch.

### Static asset paths

Because `static/` is the Vercel project root, reference assets relative
to that directory. For example, in `index.html`:

``` html
<link rel="stylesheet" href="./styles.css">
<script src="./script.js" defer></script>
```

Do not prefix these paths with `/static/` when `static` is configured as
the Vercel root directory.

Record the deployed frontend origin. It will be needed for the backend
CORS configuration.

------------------------------------------------------------------------

## Backend Deployment on Render

The backend is a Python application built with FastAPI. Render runs it
as a web service using Uvicorn.

### 1. Review the Render Blueprint

The repository contains a root-level `render.yaml` file. Review its
settings before creating the service. The configuration should:

-   Use the Python runtime.
-   Install dependencies from `requirements.txt`.
-   Install any required NLTK resources.
-   Run `python scripts/train.py` if model artifacts are generated as
    part of deployment.
-   Start the API with Uvicorn.
-   Bind to `0.0.0.0` and Render's assigned `$PORT`.
-   Configure `/health` as the health-check path.

The start command should be equivalent to:

``` bash
uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

The training command should only be included if the training script is
suitable to run during a deployment build and reliably creates the files
needed by the API.

### 2. Create a new Render service

1.  Sign in to [Render](https://dashboard.render.com/).
2.  Select **New → Blueprint**.
3.  Connect the GitHub repository
    `ananya3012/support-ticket-classifier`.
4.  Select the `main` branch.
5.  Allow Render to detect the root-level `render.yaml`.
6.  Review the service name, region, plan, build command, start command
    and environment variables.
7.  Apply the Blueprint to create the service.

This process creates a new Render service from the configuration in the
repository. It does not require an existing Render service.

### 3. Monitor the first deployment

Open the new service in the Render dashboard and inspect its deployment
events and logs.

Check for the following:

-   Python dependencies install successfully.
-   Required NLTK resources are available.
-   Model training completes, if configured as part of the build.
-   Model artifacts are saved to the expected paths.
-   Uvicorn starts without import or startup errors.
-   The `/health` health check succeeds.

If the build or startup fails, use the first relevant error in the
Render logs to identify the cause before changing deployment settings.

### 4. Record the backend URL

After the service is created, Render provides a public URL similar to:

``` text
https://your-service-name.onrender.com
```

Use the actual URL assigned in the Render dashboard. Do not assume that
an example URL or an old URL is active.

------------------------------------------------------------------------

## Configure Frontend-to-Backend Communication

### 1. Set the frontend API URL

In `static/script.js`, configure the API base URL to the actual Render
service URL:

``` javascript
const API_URL = "https://YOUR-RENDER-SERVICE.onrender.com";
```

Replace the placeholder with the URL assigned by Render. Do not use
`localhost` in the deployed frontend.

Commit and push the change to GitHub. Vercel should then create a new
deployment from the updated frontend code.

### 2. Configure CORS in FastAPI

The backend must allow browser requests from the deployed Vercel
frontend origin.

Configure the FastAPI `CORSMiddleware` in `src/api/main.py` to allow the
exact frontend origin. A Render environment variable such as
`FRONTEND_ORIGIN` can be used to supply that origin.

Example:

``` python
import os
from fastapi.middleware.cors import CORSMiddleware

frontend_origin = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5500"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Set `FRONTEND_ORIGIN` in the Render service's environment settings to
the deployed Vercel origin, for example:

``` text
https://your-frontend.vercel.app
```

Use the exact origin, without a trailing slash. If you use a custom
domain, add that origin as well. For local development, configure the
local frontend origin separately.

Do not use a wildcard origin with credentialed browser requests.

------------------------------------------------------------------------

## Verify the Deployment

Use the actual Render URL for all backend checks.

### 1. Health check

Open:

``` text
https://YOUR-RENDER-SERVICE.onrender.com/health
```

Confirm that the endpoint returns a successful HTTP response and that
the response indicates whether the model is loaded, if that information
is included by the application.

### 2. Interactive API documentation

Open:

``` text
https://YOUR-RENDER-SERVICE.onrender.com/docs
```

Use Swagger UI to inspect the available endpoints and test a prediction
request.

### 3. Test ticket prediction

Send a request to `POST /predict` using the request schema defined by
the application. For example, if the API schema uses `subject` and
`description`:

``` bash
curl -X POST "https://YOUR-RENDER-SERVICE.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Cannot log in to my account",
    "description": "I receive an invalid credentials error when trying to log in."
  }'
```

Confirm that the response contains the prediction fields defined by the
API.

### 4. Test the live frontend

1.  Open the deployed Vercel website.
2.  Submit a sample support ticket.
3.  Confirm that the browser sends a request to the Render API.
4.  Confirm that the category and priority results are displayed.
5.  Check the browser developer console and Network panel for failed
    requests or CORS errors.

### Deployment checklist

-   [x] Static frontend deployed on Vercel
-   [ ] New FastAPI backend service created on Render
-   [ ] Render build and startup complete successfully
-   [ ] Model artifacts generated and loaded by the API
-   [ ] `/health` endpoint verified
-   [ ] `/docs` and `/predict` endpoints tested
-   [ ] Frontend API URL updated to the actual Render URL
-   [ ] CORS configured for the Vercel frontend origin
-   [ ] End-to-end ticket classification tested on the live website

Update this checklist only after each item has been verified.

------------------------------------------------------------------------

## Troubleshooting

### Render build fails

-   Check that all required packages are listed in `requirements.txt`.
-   Check the Python version and dependency compatibility.
-   Inspect the build logs for NLTK download or model-training errors.
-   Confirm that the training script writes model artifacts to the paths
    expected by the API.

### Render service starts but health check fails

-   Confirm that the start command points to `src.api.main:app`.
-   Confirm that the application binds to `0.0.0.0` and uses `$PORT`.
-   Check the startup logs for import errors, missing files or
    model-loading failures.
-   Confirm that `/health` is implemented and returns a successful
    response.

### Frontend cannot reach the backend

-   Confirm that `API_URL` contains the correct Render URL.
-   Confirm that the Render service is running.
-   Check the browser Network panel for the failed request and HTTP
    status.
-   Check that the Vercel origin is included in the backend CORS
    configuration.
-   Confirm that the frontend is not still requesting `localhost`.

### First request is slow

Render free web services may spin down after a period without traffic.
The first request after inactivity can take longer while the service
starts. Check the current Render plan details for its behavior and
limits.

------------------------------------------------------------------------

## Local Development

For local development, run the API from the repository root:

``` bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Serve the static frontend in a second terminal:

``` bash
python -m http.server 5500 --directory static
```

Open:

``` text
http://127.0.0.1:5500
```

For local development, the frontend API base URL should point to:

``` javascript
const API_URL = "http://127.0.0.1:8000";
```

For the deployed frontend, use the actual Render service URL instead.

------------------------------------------------------------------------

## Useful Links

-   [Vercel Documentation](https://vercel.com/docs)
-   [Render Documentation](https://render.com/docs)
-   [FastAPI Documentation](https://fastapi.tiangolo.com/)
-   [Uvicorn Documentation](https://www.uvicorn.org/)
