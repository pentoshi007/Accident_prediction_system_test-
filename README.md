# AI-Based Accident Hotspot Prediction System — India

A proactive road-safety intelligence platform combining **GIS spatial analysis** with a hybrid ML pipeline (DBSCAN + Random Forest) to identify accident Black Spots across India and predict severity under varying road and environmental conditions.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Clone the Repository](#step-1--clone-the-repository)
4. [Step 2 — Backend Setup](#step-2--backend-setup)
5. [Step 3 — Frontend Setup](#step-3--frontend-setup)
6. [Step 4 — Start the Servers](#step-4--start-the-servers)
7. [Step 5 — Upload Data & Run the Pipeline](#step-5--upload-data--run-the-pipeline)
8. [Verify Everything Works](#verify-everything-works)
9. [Restarting / Resetting](#restarting--resetting)
10. [ML Pipeline Details](#ml-pipeline-details)
11. [API Reference](#api-reference)
12. [Configuration](#configuration)
13. [Project Structure](#project-structure)

---

## Architecture Overview

```
road.csv (Road Accident Data)
    │
    ▼  Uploaded via UI → POST /api/upload → POST /api/pipeline/run
┌──────────────────────────────────────────────────────────────────┐
│  ML Pipeline (5 stages)                                          │
│  1. Preprocess  → cleaned CSV + label encoders                   │
│  2. EDA         → 11 JSON stat files                             │
│  3. DBSCAN      → geographic clusters (India cities)             │
│  4. RandomForest→ severity classifier (~84% accuracy)            │
│  5. ARI         → risk scores per cluster                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │  .joblib models + EDA JSON files
                             ▼
                  ┌──────────────────────┐
                  │  Flask REST API      │  ← python app.py (port 5000)
                  └──────────┬───────────┘
                             │  /api proxy (Vite)
                             ▼
                  ┌──────────────────────┐
                  │  React SPA (Vite)    │  ← npm run dev (port 5173)
                  │  Dashboard · Map     │
                  │  Analytics · Predict │
                  │  Data Manager        │
                  └──────────────────────┘
```

---

## Prerequisites

Install these before you begin:

| Tool    | Minimum version | Check command         |
|---------|-----------------|-----------------------|
| Python  | 3.10+           | `python3 --version`   |
| Node.js | 18+             | `node --version`      |
| npm     | 9+              | `npm --version`       |
| git     | any             | `git --version`       |

---

## Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd Accident_prediction_system_test-
```

---

## Step 2 — Backend Setup

All commands run from the `backend/` directory.

```bash
# 1. Enter the backend directory
cd backend

# 2. Create a Python virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
#    macOS / Linux:
source venv/bin/activate
#    Windows (Command Prompt):
# venv\Scripts\activate.bat
#    Windows (PowerShell):
# venv\Scripts\Activate.ps1

# 4. Install Python dependencies
pip install -r requirements.txt
```

You should see your prompt prefixed with `(venv)` after activation.

---

## Step 3 — Frontend Setup

Open a **new terminal** and run from the project root:

```bash
# Enter the frontend directory
cd frontend

# Install Node dependencies
npm install
```

---

## Step 4 — Start the Servers

You need **two terminals running simultaneously**.

### Terminal 1 — Flask API (Backend)

```bash
cd backend
source venv/bin/activate     # skip if already active
python app.py
```

Expected output:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

Confirm it is healthy:
```bash
curl http://localhost:5000/api/health
```
Expected (before pipeline runs — models are `false` until data is uploaded):
```json
{"status": "ok", "models_loaded": {"ari_data": false, "random_forest": false, ...}}
```

### Terminal 2 — Vite Dev Server (Frontend)

```bash
cd frontend
npm run dev
```

Expected output:
```
  VITE v5.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173** in your browser.

> All `/api/*` requests are automatically proxied from port 5173 → 5000 by Vite. No CORS setup needed.

---

## Step 5 — Upload Data & Run the Pipeline

The pipeline is driven entirely from the **Data Manager** page in the UI.

1. Get the dataset:
   - Download the **RTA Dataset** from [Kaggle](https://www.kaggle.com) (Ethiopian Road Traffic Accident dataset)
   - The file should be named `road.csv`

2. Open **http://localhost:5173/data** in your browser

3. Click **Choose File**, select `road.csv`, then click **Upload**
   - You will see a success banner and a new entry appear in the Upload History

4. Click **Run Pipeline**
   - The pipeline runs all 5 stages (typically 30–90 seconds)
   - The history entry updates to **Complete** with metrics: records processed, clusters found, model accuracy, ARI range

5. Navigate to the other pages — all data is now live:
   - **Dashboard** → cluster KPIs and incident table
   - **Hotspot Map** → circle markers at Indian city coordinates
   - **Analytics** → EDA charts (hourly, weekly, severity, weather, etc.)
   - **Predict** → real-time severity + ARI prediction form

### Alternatively — run the pipeline from the command line

If you prefer not to use the UI:

```bash
cd backend
source venv/bin/activate

# Copy your dataset first
cp /path/to/road.csv data/road.csv

# Run all 5 pipeline stages
python run_pipeline.py
```

---

## Verify Everything Works

After the pipeline completes, run these smoke tests:

```bash
# Health check — all models should now be true
curl http://localhost:5000/api/health

# Clusters GeoJSON — should show N clusters
curl http://localhost:5000/api/clusters | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(len(d['features']), 'clusters')"

# EDA hourly distribution
curl http://localhost:5000/api/eda/hourly

# Upload history
curl http://localhost:5000/api/uploads

# Test a prediction (cluster 1, rainy evening)
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"cluster_id": 1, "hour": 18, "day_of_week": 4, "weather": "Rain", "num_vehicles": 2}'
```

| Page         | URL                              | What to check                                    |
|--------------|----------------------------------|--------------------------------------------------|
| Dashboard    | http://localhost:5173/           | Cluster count, incident counts, ARI scores       |
| Hotspot Map  | http://localhost:5173/map        | Circle markers at Indian city locations          |
| Analytics    | http://localhost:5173/analytics  | EDA charts populated with real data              |
| Predict      | http://localhost:5173/predict    | Cluster dropdown populated; predictions work     |
| Data Manager | http://localhost:5173/data       | Upload history entry shows Complete + metrics    |

---

## Restarting / Resetting

### Restart only the backend

```bash
# Find and kill the process on port 5000
lsof -ti :5000 | xargs kill -9

# Start it again
cd backend
source venv/bin/activate
python app.py
```

### Restart only the frontend

```bash
# Stop with Ctrl+C in the Vite terminal, then:
cd frontend
npm run dev
```

### Full reset — wipe all data and start fresh

```bash
cd backend

# Remove all generated files
rm -f data/road.csv
rm -f data/processed_accidents.csv
rm -f data/upload_history.json
rm -rf data/eda
rm -f models/*.joblib

# Restart backend
lsof -ti :5000 | xargs kill -9
source venv/bin/activate
python app.py
```

After a full reset, go to **Data Manager** and upload your CSV again to rebuild everything.

---

## ML Pipeline Details

### DBSCAN Clustering

| Parameter     | Value     | Notes                                                      |
|---------------|-----------|------------------------------------------------------------|
| `eps`         | 0.005 rad | ~32 km radius — groups accidents within a city, separates cities |
| `min_samples` | 15        | Min accidents required to form a Black Spot cluster        |
| `metric`      | haversine | Geographically accurate great-circle distances             |
| `algorithm`   | ball_tree | Efficient spatial indexing for haversine                   |

### Random Forest features (17)

```
Hour, DayOfWeek, Is_Night, Weather_Binned_Enc, Num_Vehicles,
Type_of_vehicle_Enc, Road_surface_type_Enc, Road_surface_conditions_Enc,
Light_conditions_Enc, Type_of_collision_Enc, Cause_of_accident_Enc,
Road_allignment_Enc, Types_of_Junction_Enc, Lanes_or_Medians_Enc,
Driving_experience_Enc, Age_band_of_driver_Enc, Cluster_ID
```

`Cluster_ID` is included so the model is spatially aware — severity patterns differ between city types.

### Accident Risk Index

```
ARI = W1 × Severity_Score + W2 × Density_Score + W3 × Environmental_Factor
```

| Component             | Derivation                                          |
|-----------------------|-----------------------------------------------------|
| Severity_Score        | Mean severity in cluster, normalised to [0, 1]      |
| Density_Score         | Cluster incident count / max count across clusters  |
| Environmental_Factor  | Weather risk score (Clear = 0.15 … Snow = 0.85)    |
| W1, W2, W3            | Derived from RF feature importances (sum to 1)      |

| Risk Tier | ARI Range   | Map colour |
|-----------|-------------|------------|
| Low       | 0.00 – 0.30 | Green      |
| Moderate  | 0.30 – 0.50 | Yellow     |
| Severe    | 0.50 – 0.70 | Orange     |
| Critical  | 0.70 – 1.00 | Red        |

### Geocoding — area type → Indian city

| Area type           | Mapped city      |
|---------------------|------------------|
| Office areas        | Delhi NCR        |
| Residential areas   | Mumbai           |
| Church areas        | Bangalore        |
| Industrial areas    | Ahmedabad        |
| School areas        | Chennai          |
| Recreational areas  | Kolkata          |
| Hospital areas      | Jaipur           |
| Market areas        | Hyderabad        |
| Rural village areas | Nagpur           |
| Outside rural areas | Varanasi         |
| Other               | Pune             |
| Unknown             | Centre of India  |

---

## API Reference

### Core

| Method | Endpoint                    | Description                                     |
|--------|-----------------------------|-------------------------------------------------|
| GET    | `/api/health`               | Server and model load status                    |
| GET    | `/api/clusters`             | All clusters as GeoJSON FeatureCollection       |
| GET    | `/api/clusters/<id>`        | Single cluster detail                           |
| POST   | `/api/predict`              | Severity + ARI prediction for given conditions  |

### EDA

| Method | Endpoint                       | Returns                          |
|--------|--------------------------------|----------------------------------|
| GET    | `/api/eda/hourly`              | `[{hour, count}]` × 24          |
| GET    | `/api/eda/weekly`              | `[{day, count}]` × 7            |
| GET    | `/api/eda/severity`            | `[{label, count}]`              |
| GET    | `/api/eda/weather`             | `[{weather, count}]`            |
| GET    | `/api/eda/top_areas`           | `[{area, count}]`               |
| GET    | `/api/eda/collision_types`     | `[{collision_type, count}]`     |
| GET    | `/api/eda/causes`              | `[{cause, count}]`              |
| GET    | `/api/eda/severity_by_weather` | Cross-tab array                 |
| GET    | `/api/model/metrics`           | Accuracy, confusion matrix, feature importances |

### Upload & Pipeline

| Method | Endpoint                  | Description                                                 |
|--------|---------------------------|-------------------------------------------------------------|
| POST   | `/api/upload`             | Upload a CSV file (multipart/form-data, field name: `file`) |
| POST   | `/api/pipeline/run`       | Run full ML pipeline (blocking, ~30–90 s)                   |
| GET    | `/api/uploads`            | Full upload history (newest first)                          |
| DELETE | `/api/uploads/<id>`       | Remove a history entry                                      |
| DELETE | `/api/uploads/<id>?clear_artifacts=true` | Remove history entry + wipe all model and EDA files |

### Prediction request body

```json
{
  "cluster_id": 1,
  "hour": 17,
  "day_of_week": 4,
  "is_night": 1,
  "weather": "Rain",
  "num_vehicles": 2,
  "road_surface_conditions": "Wet or damp",
  "light_conditions": "Darkness - lights unlit",
  "type_of_collision": "Vehicle with vehicle collision",
  "cause_of_accident": "No distancing",
  "driving_experience": "5-10yr",
  "age_band_of_driver": "18-30"
}
```

`cluster_id`, `hour`, and `weather` are required. All other fields are optional.

### Prediction response

```json
{
  "predicted_label": "Slight Injury",
  "severity_probabilities": {
    "Slight Injury": 0.8292,
    "Serious Injury": 0.1431,
    "Fatal Injury": 0.0277
  },
  "ari_score": 0.73,
  "risk_tier": "Critical",
  "weights": {
    "W1_severity": 0.54,
    "W2_density": 0.39,
    "W3_environment": 0.07
  }
}
```

---

## Configuration

All config lives in `backend/config.py`. Override with environment variables:

| Variable             | Default   | Description                          |
|----------------------|-----------|--------------------------------------|
| `DBSCAN_EPS`         | `0.005`   | Cluster radius in radians (~32 km)   |
| `DBSCAN_MIN_SAMPLES` | `15`      | Min accidents to form a cluster      |
| `RF_N_ESTIMATORS`    | `200`     | Number of trees in Random Forest     |
| `RF_MAX_DEPTH`       | `20`      | Maximum tree depth                   |
| `RF_TEST_SIZE`       | `0.2`     | Test set fraction (stratified split) |

---

## Project Structure

```
Accident_prediction_system_test-/
├── README.md
├── backend/
│   ├── app.py                       # Flask application entry point
│   ├── config.py                    # Centralised paths & hyperparameters
│   ├── requirements.txt             # Python dependencies
│   ├── run_pipeline.py              # CLI pipeline runner (alternative to UI)
│   ├── data/
│   │   ├── road.csv                 # Raw dataset (place here or upload via UI)
│   │   ├── processed_accidents.csv  # Generated by pipeline step 1
│   │   ├── upload_history.json      # Upload history (auto-created)
│   │   └── eda/                     # Generated EDA JSON files
│   ├── models/                      # Serialised ML artefacts (generated)
│   │   ├── rf_model.joblib
│   │   ├── dbscan_labels.joblib
│   │   ├── cluster_data.joblib
│   │   ├── ari_data.joblib
│   │   ├── label_encoders.joblib
│   │   └── feature_importances.joblib
│   ├── scripts/                     # Pipeline stages
│   │   ├── preprocess.py            # Step 1 — clean + encode
│   │   ├── eda.py                   # Step 2 — stat distributions
│   │   ├── clustering.py            # Step 3 — DBSCAN
│   │   ├── classifier.py            # Step 4 — Random Forest
│   │   └── ari.py                   # Step 5 — risk index
│   ├── routes/                      # Flask API blueprints
│   │   ├── clusters.py
│   │   ├── predictions.py
│   │   ├── eda_routes.py
│   │   └── upload.py                # Upload, pipeline, history endpoints
│   └── utils/
│       └── geojson_utils.py         # Cluster → GeoJSON conversion
└── frontend/
    ├── package.json
    ├── vite.config.js               # /api proxy → port 5000
    └── src/
        ├── api.js                   # Axios wrappers for all API calls
        ├── App.jsx                  # Layout + routing
        ├── ThemeContext.jsx         # Dark / light theme toggle
        ├── components/              # Shared UI components
        └── pages/
            ├── Dashboard.jsx        # KPI overview + cluster table
            ├── MapView.jsx          # Leaflet hotspot map
            ├── Analytics.jsx        # EDA charts
            ├── Predict.jsx          # Real-time severity prediction
            └── DataManager.jsx      # Upload, pipeline, history, CSV guide
```
