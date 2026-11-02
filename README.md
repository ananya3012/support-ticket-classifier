# 🎫 Support Ticket Classification Platform

[![Python
3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[\![scikit-learn\](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg](https://scikit-learn.org/)

An AI/ML-based support ticket classification platform that predicts
ticket categories and priorities from ticket subject and description. It
provides a browser-based interface and a RESTful API built with FastAPI.

## Live Application

-   **Frontend:** `https://support-ticket-classifier-static.vercel.app`
-   **Backend API:** `ADD_YOUR_RENDER_URL_HERE` *(add after backend
    deployment)*
-   **Interactive API documentation:** `ADD_YOUR_RENDER_URL_HERE/docs`
    *(available after backend deployment)*

> The frontend is deployed separately from the API. The backend
> deployment is not considered complete until the Render service is live
> and the frontend successfully communicates with it.

## Features

-   **Ticket category prediction:** predicts a category for an
    individual support ticket.
-   **Priority prediction:** predicts a priority level for an individual
    ticket.
-   **Confidence scores:** returns confidence values and class
    probability information where provided by the model/API.
-   **Batch prediction:** supports submitting multiple tickets through
    the batch endpoint.
-   **API health check:** exposes a health endpoint for service checks.
-   **Model metrics:** exposes a metrics endpoint.

The categories, priority labels, response fields and behavior are
defined by the implementation in the source code.

## Technology Stack

  Component                   Technology
  --------------------------- ----------------------------
  Frontend                    HTML, CSS, JavaScript
  Backend framework           Python, FastAPI
  ASGI server                 Uvicorn
  Data validation / schemas   Pydantic
  Machine learning            scikit-learn
  Text processing             NLTK
  Data handling               Pandas, NumPy
  Model serialization         Joblib
  API style                   RESTful HTTP API with JSON
  Frontend hosting            Vercel
  Backend hosting             Render

The repository also contains Docker-related files and
testing/visualization components. These are project capabilities and
local-development options; they should not be interpreted as evidence
that the application is deployed on AWS or running in Docker in
production.

## Project Structure

``` text
support-ticket-classifier/
├── src/
│   ├── api/                 # FastAPI application and API schemas
│   ├── data/                # Data generation, database and preprocessing
│   ├── models/              # Classification model and related code
│   └── visualizations/      # Model/data visualizations
├── scripts/
│   └── train.py             # Model training pipeline
├── static/
│   ├── index.html           # Frontend page
│   ├── script.js            # Frontend behavior and API requests
│   └── styles.css           # Frontend styling
├── tests/                   # Automated tests
├── docs/                    # Additional project documentation
├── requirements.txt         # Python dependencies
├── render.yaml              # Render Blueprint configuration
├── Dockerfile               # Docker image configuration
├── docker-compose.yml       # Docker Compose configuration
└── README.md
```

> The structure above highlights the main application files. Other files
> and directories may also be present in the repository.

## Local Setup

### Prerequisites

-   Python 3.11 or a compatible version supported by the dependencies
-   Git
-   pip

### 1. Clone the repository

``` bash
git clone https://github.com/ananya3012/support-ticket-classifier.git
cd support-ticket-classifier
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Download NLTK resources

Run the following command if the preprocessing pipeline requires these
NLTK resources:

``` bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Train the model

``` bash
python scripts/train.py
```

Run the training pipeline before starting the API if the model artifacts
required by the application are not already present. The training script
is responsible for generating/training and saving the artifacts
according to its implementation.

### 6. Start the backend API

From the repository root:

``` bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

The API should then be available at:

-   API base URL: `http://127.0.0.1:8000`
-   Swagger UI: `http://127.0.0.1:8000/docs`
-   ReDoc: `http://127.0.0.1:8000/redoc`
-   Health check: `http://127.0.0.1:8000/health`

### 7. Run the frontend locally

The frontend files are in `static/`. You can serve that directory with a
local static file server, for example:

``` bash
python -m http.server 5500 --directory static
```

Then open `http://127.0.0.1:5500` in your browser.

Ensure the frontend API base URL in `static/script.js` points to the
local backend while developing:

``` javascript
const API_URL = "http://127.0.0.1:8000";
```

If you change the frontend API URL for local development, restore the
deployed Render URL before committing a production frontend change.

## REST API

The backend is implemented with **FastAPI** and exposes HTTP endpoints.
FastAPI generates an OpenAPI schema and interactive documentation for
the API.

### Endpoints

  Method   Endpoint           Purpose
  -------- ------------------ ------------------------------------------------
  `GET`    `/health`          Health check
  `GET`    `/metrics`         Model metrics
  `POST`   `/predict`         Predict the category and priority for a ticket
  `POST`   `/predict/batch`   Predict for multiple tickets
  `GET`    `/categories`      Retrieve supported categories
  `GET`    `/priorities`      Retrieve supported priority levels

The exact request and response schemas are defined in
`src/api/schemas.py` and the route handlers in `src/api/main.py`.

### Example request

``` bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Cannot log in to my account",
    "description": "I have been trying to log in but receive an invalid credentials error."
  }'
```

### API documentation

When the backend is running:

-   Swagger UI: `/docs`
-   ReDoc: `/redoc`
-   OpenAPI schema: `/openapi.json`

For a deployed service, replace the local base URL with the Render
service URL.

## Machine Learning

The project README and implementation describe a supervised
text-classification workflow using TF-IDF features and Logistic
Regression.

At a high level, the workflow is:

1.  Prepare ticket text and labels.
2.  Preprocess the text using the project's NLP pipeline.
3.  Convert text into numerical features using TF-IDF.
4.  Train classification model(s).
5.  Serialize the trained model artifacts for inference.
6.  Load the artifacts in the API to serve predictions.

The training and preprocessing implementation is maintained in
`scripts/train.py` and the relevant modules under `src/`. Model
performance depends on the training data and evaluation procedure;
results should be reported using the latest measured metrics rather than
assumed or illustrative figures.

## Deployment

The application uses a separated frontend/backend deployment
architecture.

### Frontend --- Vercel

The static frontend is located in `static/` and is deployed on Vercel.

For the Vercel project configuration:

  Setting            Value
  ------------------ ------------------------------------
  Framework Preset   Other
  Root Directory     `static`
  Build Command      Leave empty for plain static files
  Output Directory   Leave empty / default

The frontend uses `static/index.html`, `static/styles.css` and
`static/script.js`.

### Backend --- Render

The FastAPI backend is intended to be deployed as a Render Web Service
using the repository's root-level `render.yaml` Blueprint configuration.

The service configuration should:

1.  Install dependencies from `requirements.txt`.
2.  Install/download any required NLP resources.
3.  Run `scripts/train.py` if the model artifacts are generated during
    deployment.
4.  Start the application with Uvicorn, binding to `0.0.0.0` and
    Render's `$PORT`.
5.  Use `/health` as the health-check endpoint.

Review `render.yaml` before deploying and ensure its build/start
commands match the current codebase. A successful build does not by
itself prove that model loading or prediction works; verify the service
logs and test the API endpoints after deployment.

### Connect the frontend to the backend

After Render creates the service, copy its assigned URL and set the API
base URL in `static/script.js`:

``` javascript
const API_URL = "https://YOUR-RENDER-SERVICE.onrender.com";
```

Replace the example with the actual Render URL. Do not use `localhost`
in the deployed frontend.

### Configure CORS

The FastAPI backend must allow requests from the deployed Vercel
frontend origin. Configure the backend's CORS middleware using the exact
Vercel origin (without a trailing slash), preferably through a Render
environment variable such as `FRONTEND_ORIGIN`.

For local development, allow the local frontend origin separately as
appropriate. Avoid using a wildcard origin when credentialed requests
are enabled.

### Deployment status

-   [x] Static frontend deployed on Vercel
-   [ ] FastAPI backend deployed on Render
-   [ ] Production frontend connected to the deployed backend
-   [ ] Production API health and prediction endpoints verified

Update this checklist and the Live Application links after completing
and testing the backend deployment.

## Testing

Install the test dependencies from `requirements.txt`, then run:

``` bash
pytest tests/ -v
```

To run a specific test module:

``` bash
pytest tests/test_api.py -v
```

The available test modules and coverage depend on the files currently
present under `tests/`.

## Docker (Optional Local Deployment)

The repository includes a `Dockerfile` and `docker-compose.yml`. If you
want to run the application in Docker, first review those files and
ensure their commands, paths and required model artifacts match the
current project structure.

Example commands:

``` bash
docker build -t support-ticket-classifier .
docker run --rm -p 8000:8000 support-ticket-classifier
```

Docker support in the repository does not mean the deployed
Vercel/Render application is hosted using Docker.

## Future Improvements

Potential areas for further development include:

-   Evaluating and improving classification performance with a larger,
    representative dataset.
-   Experimenting with transformer-based text-classification models.
-   Adding sentiment analysis.
-   Supporting additional languages.
-   Introducing an active-learning workflow.
-   Adding monitoring and a model-retraining pipeline.

## Contributing

1.  Fork the repository.
2.  Create a feature branch: `git checkout -b feature/your-feature`.
3.  Commit your changes: `git commit -m "Describe your change"`.
4.  Push the branch: `git push origin feature/your-feature`.
5.  Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file
for details.

## Author

**Ananya Gupta**

------------------------------------------------------------------------

Made with ❤️ using Python, FastAPI, scikit-learn, HTML, CSS and
JavaScript.
