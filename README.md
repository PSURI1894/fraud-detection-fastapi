# Fraud Detection System (LightGBM + FastAPI + Vercel)

End‑to‑end fraud detection project with:

- **Offline training** using LightGBM and scikit‑learn pipelines
- **Real‑time inference API** using FastAPI
- **Containerized deployment** with Docker
- **Serverless deployment** on **Vercel** (Python Runtime + ASGI)

This is designed as a portfolio‑ready project that shows you can take a model
from notebook → production API.

---

## 1. Project Structure

```text
fraud-detection-fastapi/
├─ app/
│  ├─ __init__.py
│  ├─ main.py                # FastAPI app (real-time inference API)
│  └─ models/
│     └─ (lgbm_model.pkl)    # Trained LightGBM pipeline (created after training)
├─ training/
│  ├─ __init__.py
│  └─ train.py               # Offline training script
├─ data/
│  └─ transactions.csv       # Your fraud dataset (NOT committed; local only)
├─ api/
│  └─ index.py               # Vercel entrypoint that exposes the FastAPI app
├─ requirements.txt
├─ Dockerfile
└─ vercel.json
```

- `train.py` trains a LightGBM model and saves it as `app/models/lgbm_model.pkl`.
- `app/main.py` exposes a FastAPI app that loads the saved model and serves `/predict`.
- `api/index.py` re‑exports the same FastAPI app for Vercel's Python Runtime.

---

## 2. Tech Stack

- **Python 3.10+**
- **FastAPI** for the REST API
- **LightGBM** for fraud detection model
- **scikit‑learn** for preprocessing and evaluation
- **Pydantic** for request/response schemas
- **Uvicorn** for local development server
- **Docker** (optional) for containerized deployment
- **Vercel** for serverless deployment

---

## 3. Dataset Expectations

Place your CSV dataset as:

```text
data/transactions.csv
```

The training script expects a binary target column named:

- `is_fraud` — 0 for genuine, 1 for fraudulent

All other columns are treated as features. Numeric columns are standardized;
categorical columns are one‑hot encoded. You can customize this in
`training/train.py` if your schema is different.

Example minimal schema (you can have more features):

- `amount` (float)
- `oldbalanceOrg` (float)
- `newbalanceOrig` (float)
- `transaction_type` (string; e.g., `CASH_OUT`, `PAYMENT`, ...)
- `is_fraud` (0/1)

---

## 4. Setup & Installation

### 4.1. Clone and create a virtual environment

```bash
git clone <your-repo-url> fraud-detection-fastapi
cd fraud-detection-fastapi

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2. Add your dataset

```bash
mkdir -p data
# Place your transactions.csv in ./data
```

---

## 5. Training the LightGBM Model

The training script:

- Loads `data/transactions.csv`
- Splits into train/validation
- Builds a preprocessing pipeline:
  - `StandardScaler` for numeric features
  - `OneHotEncoder` for categorical features
- Trains a `LGBMClassifier`
- Evaluates using ROC‑AUC and prints a classification report
- Saves the fitted pipeline (preprocessing + model) to `app/models/lgbm_model.pkl`

Run:

```bash
python training/train.py
```

After it finishes, you should see:

```text
app/models/lgbm_model.pkl
```

committed to your local file system. For deployment to Vercel, you will **commit
this model file** into your repository so Vercel can load it at runtime.

> In a real production setup, you would usually load the model from a model
> registry or object storage. For a portfolio project, committing the artifact
> is perfectly fine.

---

## 6. Running the API Locally (Uvicorn)

Once the model is trained and saved, run the FastAPI app locally:

```bash
uvicorn app.main:app --reload
```

- API root:      `http://127.0.0.1:8000/`
- Health check:  `http://127.0.0.1:8000/health`
- Docs (Swagger):`http://127.0.0.1:8000/docs`
- OpenAPI JSON:  `http://127.0.0.1:8000/openapi.json`

### 6.1. Example request to `/predict`

`POST /predict`

```json
{
  "amount": 1000.0,
  "oldbalanceOrg": 5000.0,
  "newbalanceOrig": 4000.0,
  "transaction_type": "CASH_OUT"
}
```

Example response:

```json
{
  "is_fraud": 0,
  "fraud_probability": 0.08342194557189941
}
```

- `is_fraud`: model's decision (0 = genuine, 1 = fraud)
- `fraud_probability`: probability of fraud from the model (0.0–1.0)

---

## 7. API Overview

### 7.1. `GET /health`

Return a basic health status for uptime checks.

```json
{
  "status": "ok",
  "message": "Fraud Detection API is running"
}
```

### 7.2. `POST /predict`

Request body (Pydantic model: `TransactionInput`):

- `amount`: float
- `oldbalanceOrg`: float
- `newbalanceOrig`: float
- `transaction_type`: string

Response body (Pydantic model: `FraudPredictionResponse`):

- `is_fraud`: int (0/1)
- `fraud_probability`: float

---

## 8. Docker Deployment (Optional)

You can build and run the app in a Docker container.

```bash
# Build image
docker build -t fraud-fastapi .

# Run container
docker run -p 8000:8000 fraud-fastapi
```

The API will be available at `http://localhost:8000` (same endpoints as local run).

> Note: The Docker image assumes that `app/models/lgbm_model.pkl` already exists.
> Run `python training/train.py` **before** building the image and commit the
> generated model file if you're building in a CI pipeline.

---

## 9. Vercel Deployment (Serverless)

This project is configured to be deployable to **Vercel** using its Python
Runtime and ASGI support for FastAPI.

### 9.1. How the Vercel entrypoint works

Vercel expects a Python file under the `api/` directory that exposes an ASGI
application variable named `app`. In this project:

- `app/main.py` defines the main FastAPI app (`app = FastAPI(...)`).
- `api/index.py` simply imports and re‑exports this app:

  ```python
  from app.main import app  # FastAPI instance
  ```

Vercel will pick up `api/index.py` and run the `app` ASGI application.

### 9.2. Prepare the repository for deployment

1. Train the model locally:

   ```bash
   python training/train.py
   ```

2. Ensure the model file exists:

   ```text
   app/models/lgbm_model.pkl
   ```

3. Commit the code **and** the model file to your Git repository
   (for a portfolio/demo project).

   ```bash
   git add .
   git commit -m "Initial fraud detection API"
   git push origin main
   ```

### 9.3. Create a Vercel project (dashboard)

1. Go to the Vercel dashboard.
2. Click **New Project**.
3. Import your GitHub repository.
4. Vercel will detect `api/index.py` and configure a Python FastAPI backend.
5. Click **Deploy**.

Once deployment succeeds, you will get a URL like:

```text
https://your-fraud-api.vercel.app
```

API endpoints will be available as:

- Health:  `https://your-fraud-api.vercel.app/api/health`
- Docs:    `https://your-fraud-api.vercel.app/api/docs`
- Predict: `https://your-fraud-api.vercel.app/api/predict`

(Vercel automatically mounts Python functions under the `/api` prefix.)

### 9.4. Deploy via Vercel CLI (optional)

```bash
npm install -g vercel

# From the project root
vercel login
vercel
```

Follow the interactive prompts. Subsequent deployments can be done with:

```bash
vercel --prod
```

---

## 10. Environment Notes

- If you later add environment variables (e.g., external model registry, DB
  connection), configure them in the Vercel dashboard under **Settings →
  Environment Variables**.
- The current setup reads only local files (`app/models/lgbm_model.pkl`),
  so no environment variables are required by default.

---

## 11. Extending the Project

Ideas to make this even stronger for your portfolio:

- Log requests and predictions (e.g., to a database or log file).
- Add a `/metrics` endpoint (e.g., Prometheus format or simple JSON stats).
- Implement simple authentication (API key header, JWT, etc.).
- Add input drift / feature distribution monitoring.
- Build a small UI (React or Streamlit) that calls this API.

---

## 12. Quick Start Summary

1. Clone repo & install dependencies.
2. Put your dataset in `data/transactions.csv`.
3. Run `python training/train.py` to create `app/models/lgbm_model.pkl`.
4. Run locally with `uvicorn app.main:app --reload`.
5. Commit everything (including the model) and deploy to Vercel.
