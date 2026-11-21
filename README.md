# Smart Mobility Demand Forecasting and Capacity Allocation

Predict hourly shared-bicycle demand using weather, time, and seasonal context — and generate capacity recommendations to optimize bicycle fleet availability across urban stations. All predictions and audit trails are persisted to MongoDB.

> **Scope Note:** This repository provides an **API-only implementation** (FastAPI + Scikit-Learn + MongoDB). A frontend/user interface is intentionally omitted at this stage and will be developed separately. Users interact with the service via Postman, direct HTTP requests, or the built-in interactive Swagger UI (`/docs`).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Technology Stack](#3-technology-stack)
4. [Dataset Description](#4-dataset-description)
5. [Selected Prediction Features](#5-selected-prediction-features)
6. [Data Preprocessing Workflow](#6-data-preprocessing-workflow)
7. [K-Means Clustering](#7-k-means-clustering)
8. [HistGradientBoosting Regression](#8-histgradientboosting-regression)
9. [Chronological Train/Test Split](#9-chronological-traintest-split)
10. [Model Evaluation & Baseline Comparison](#10-model-evaluation--baseline-comparison)
11. [Capacity Estimation Methodology](#11-capacity-estimation-methodology)
12. [Station Allocation Methodology](#12-station-allocation-methodology)
13. [FastAPI Architecture](#13-fastapi-architecture)
14. [MongoDB Integration](#14-mongodb-integration)
15. [API Endpoint Documentation](#15-api-endpoint-documentation)
16. [Postman Testing Instructions](#16-postman-testing-instructions)
17. [Local Setup Instructions](#17-local-setup-instructions)
18. [Public Deployment with ngrok](#18-public-deployment-with-ngrok)
19. [Project Limitations](#19-project-limitations)
20. [Running the Jupyter Notebook](#20-running-the-jupyter-notebook)
21. [Documentation & Model Artifacts](#21-documentation--model-artifacts)

---

## 1. Project Overview

Urban micro-mobility operators struggle with spatial and temporal demand imbalances: rush hour surges empty commuter stations while rainy weekends leave hundreds of bicycles idle. 

This project provides an end-to-end, reproducible machine learning and backend system that:
- Aggregates city-wide trip data into a continuous hourly departure time series.
- Discovers **contextual demand profiles** through unsupervised K-Means clustering.
- Forecasts **city-wide hourly bicycle departures** using a gradient-boosted regression pipeline.
- Translates predicted demand into **recommended concurrent bicycle fleet capacity** using trip duration and safety buffers.
- Distributes predicted departures across stations using empirical pickup shares.
- Persists all prediction requests, baseline comparisons, and responses into MongoDB for auditing.

---

## 2. Problem Statement

Given meteorological conditions (temperature, precipitation, wind speed, humidity, sunshine), calendar indicators (hour, day of week, season, holiday flag), and historical mobility patterns:
1. Predict total city-wide bicycle trips departing during a target hour.
2. Contextualize whether current conditions match high, moderate, or low demand patterns.
3. Compare the forecast against a naive historical weekday-hour baseline to evaluate if demand is unusually high or low.
4. Recommend actionable fleet availability adjustments (e.g., increase or reduce bicycle inventory).
5. Allocate estimated demand across individual bicycle docking stations.

---

## 3. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | `>=3.10` (tested on `3.14`) | Core execution runtime |
| **Data Processing** | Pandas, NumPy | `^2.0` / `^1.24` | Time-series aggregation, feature engineering |
| **Machine Learning** | Scikit-Learn | `^1.4` (tested on `1.8.0`) | ColumnTransformer, KMeans, HistGradientBoostingRegressor |
| **Model Persistence** | Joblib | `^1.3` | Multi-model artifact bundling and serialization |
| **Backend API** | FastAPI, Starlette | `^0.110` | High-performance asynchronous REST API |
| **Data Validation** | Pydantic | `^2.0` | Strict request schema and range validation |
| **Database** | MongoDB & PyMongo | `^4.6` | Prediction history and audit persistence |
| **Server** | Uvicorn | `^0.28` | ASGI production application server |
| **Tunneling** | ngrok | `^3.0` | Public HTTPS exposure for external client testing |

---

## 4. Dataset Description

The original dataset (`data/a.csv`) contains **467,861 raw trip records** from Bergen Bysykkel spanning January 1, 2023 to December 31, 2023 with 29 columns:

- **Timestamps:** `start_time`, `end_time`, `start_clock_time`, `end_clock_time`
- **Trip Attributes:** `duration` (seconds), `vehicle_type`
- **Station Data:** `start_station_id`, `start_station_name`, coordinates, `end_station_id`, `end_station_name`, coordinates
- **Meteorological Data:** `temperature`, `max_temperature`, `min_temperature`, `wind_speed`, `precipitation`, `humidity`, `weather` (code), `sunshine`
- **Calendar Signals:** `season`, `is_holiday`, `is_weekend`, `date`, `hour`, `day_of_week`, `month`

### Vehicle Filtering & Target Definition
The dataset contains three distinct modalities:
- `classic_bike`: 69,037 trips
- `electric_bike`: 200,015 trips
- `electric_scooter`: 198,809 trips

> [!IMPORTANT]
> This is strictly a bicycle-demand project. Electric scooter trips are filtered out. `classic_bike` and `electric_bike` trips are combined into a single hourly demand target ($269,052$ total bicycle trips).

---

## 5. Selected Prediction Features

The API accepts **12 user-supplied input fields** in the `POST /predict` request body:

| # | Field | Type | Description | Valid Range / Rules |
|---|---|---|---|---|
| 1 | `date` | string | Target date (`YYYY-MM-DD`) | Valid calendar date |
| 2 | `hour` | integer | Hour of day | `0` to `23` |
| 3 | `temperature` | float | Hourly temperature (°C) | `min_temperature` $\le$ `temperature` $\le$ `max_temperature` |
| 4 | `max_temperature` | float | Daily maximum temperature (°C) | $\ge$ `min_temperature` |
| 5 | `min_temperature` | float | Daily minimum temperature (°C) | $\le$ `max_temperature` |
| 6 | `wind_speed` | float | Wind speed (m/s) | $\ge 0$ |
| 7 | `precipitation` | float | Hourly rainfall/snowfall (mm) | $\ge 0$ |
| 8 | `humidity` | float | Relative humidity (%) | `0` to `100` |
| 9 | `weather` | integer | WMO weather code | Integer category |
| 10 | `sunshine` | float | Sunshine duration (minutes) | $\ge 0$ |
| 11 | `season` | integer | Season identifier | Integer code (`1`–`4`) |
| 12 | `is_holiday` | boolean | Public holiday indicator | `true` or `false` |

### Derived Cyclic & Temporal Features
From the date and hour, the pipeline derives cyclic trigonometric encodings to preserve circular continuity (e.g., hour 23 is adjacent to hour 0):
- $\text{hour\_sin} = \sin(2\pi \cdot \text{hour} / 24)$, $\text{hour\_cos} = \cos(2\pi \cdot \text{hour} / 24)$
- $\text{dow\_sin} = \sin(2\pi \cdot \text{dow} / 7)$, $\text{dow\_cos} = \cos(2\pi \cdot \text{dow} / 7)$
- $\text{month\_sin} = \sin(2\pi \cdot (\text{month}-1) / 12)$, $\text{month\_cos} = \cos(2\pi \cdot (\text{month}-1) / 12)$
- $\text{day\_of\_year\_sin} = \sin(2\pi \cdot (\text{doy}-1) / 365.25)$, $\text{day\_of\_year\_cos} = \cos(2\pi \cdot (\text{doy}-1) / 365.25)$
- $\text{is\_weekend} = 1$ if day of week $\in [5, 6]$ else $0$

---

## 6. Data Preprocessing Workflow

```
Raw CSV (467K trips)
  │
  ├── 1. Filter: classic_bike + electric_bike (scooters dropped)
  │
  ├── 2. Hourly Time-Series Construction (8,755 hours total)
  │        Group departures by hour; fill zero-departure hours (1,882 hours)
  │
  ├── 3. Weather Imputation Strategy:
  │        Hourly median → Daily median fallback → Linear time interpolation
  │
  ├── 4. Chronological 80/20 Train/Test Split (No shuffling)
  │
  ├── 5. K-Means Preprocessing: StandardScaler (numericals) + OneHotEncoder (categoricals)
  │        → Assign Contextual Demand Profiles 1–4
  │
  ├── 6. Supervised Preprocessing: ColumnTransformer
  │        Numericals passthrough + OneHotEncoder on weather, season, demand_profile
  │
  └── 7. Model Training: HistGradientBoostingRegressor
```

---

## 7. K-Means Clustering

- **Objective:** Unsupervised discovery of recurring atmospheric and temporal contexts.
- **Configuration:** $k=4$ clusters, `random_state=42`, `n_init=20`.
- **Naming Style:** Neutral contextual labels (`Contextual demand profile 1` through `4`), sorted by average training demand.
- **Important Distinction:** The clusters represent **contextual similarity** (e.g., warm sunny afternoon vs. freezing winter midnight), not rigid demand guarantees. The regression model uses the profile as a feature and may predict higher or lower values than the profile's training average.

---

## 8. HistGradientBoosting Regression

- **Algorithm:** `sklearn.ensemble.HistGradientBoostingRegressor`.
- **Why HistGradientBoosting:** Native support for categorical features, exceptional speed on tabular datasets, robust handling of non-linear weather interactions, and built-in L2 regularization.
- **Hyperparameters:**
  - `learning_rate`: `0.06`
  - `max_iter`: `300`
  - `max_leaf_nodes`: `20`
  - `min_samples_leaf`: `25`
  - `l2_regularization`: `2.0`
  - `early_stopping`: `True`
  - `random_state`: `42`

---

## 9. Chronological Train/Test Split

To prevent lookahead bias and accurately reflect real-world forecasting, data is split chronologically:
- **Training Period:** January 1, 2023, 04:00 to October 19, 2023, 23:00 ($7,004$ hourly records, $80\%$).
- **Held-Out Test Period:** October 20, 2023, 00:00 to December 31, 2023, 22:00 ($1,751$ hourly records, $20\%$).
- **No Random Shuffling:** Shuffling time-series data causes synthetic data leakage from future observations into past predictions.

---

## 10. Model Evaluation & Baseline Comparison

### Weekday-Hour Baseline
The benchmark model calculates the empirical mean demand for each specific `(day_of_week, hour)` combination across the training set (168 bins). It uses zero weather data and zero machine learning.

### Verified Test Period Performance (Oct 20 – Dec 31, 2023)

| Model | MAE (trips/hr) | RMSE (trips/hr) | $R^2$ Score |
|---|---|---|---|
| **HistGradientBoostingRegressor (ML)** | **7.225** | **11.166** | **0.818** |
| **Historical Weekday-Hour Baseline** | 15.050 | 22.710 | 0.246 |
| **Improvement** | **-52.0%** (lower error) | **-50.8%** (lower error) | **+0.572** |

The ML model captures weather-driven demand fluctuations that a calendar-only baseline completely misses.

---

## 11. Capacity Estimation Methodology

Operators need to know how many physical bicycles must be deployed concurrently to meet predicted hourly departures.

### Concurrency Formula
$$\text{Turnover Rate} = \frac{60}{\bar{d}} \quad (\text{trips per bicycle per hour})$$

$$\text{Estimated Concurrent Bicycles} = \frac{\hat{Y}}{\text{Turnover Rate}} = \frac{\hat{Y} \times \bar{d}}{60}$$

$$\text{Recommended Capacity} = \left\lceil \text{Estimated Concurrent Bicycles} \times (1 + \beta) \right\rceil$$

Where:
- $\hat{Y}$: Predicted hourly departures.
- $\bar{d}$: Mean bicycle trip duration ($14.68\text{ minutes}$ in the training data).
- $\beta$: Safety buffer ($0.20$, or $20\%$).

### Operational Recommendation Thresholds
- **$\ge +15\%$ above baseline:** `"Increase bicycle availability"`
- **$\le -15\%$ below baseline:** `"Reduce excess bicycle availability"`
- **Within $\pm 15\%$:** `"Maintain current bicycle availability"`

---

## 12. Station Allocation Methodology

> [!NOTE]
> **City-wide Forecast vs. Station Distribution:** The ML regression model predicts total **city-wide departures**. Station-level values are derived using empirical pickup proportions conditioned on `(day_of_week, hour)`.

For each hour, the top 8 stations are determined by:
$$\hat{y}_{s} = \hat{Y} \times \omega_{s, \text{dow}, \text{hour}}$$

Where $\omega_{s, \text{dow}, \text{hour}}$ is station $s$'s historical share of departures for that day and hour (146 conditional distributions; global fallback used if a slot has insufficient historical data).

---

## 13. FastAPI Architecture

The backend (`main.py`) provides a clean, robust REST API:
- **Lifespan Manager:** Loads the model artifact and initializes a reusable PyMongo connection pool during server startup; closes them gracefully upon shutdown.
- **Pydantic Validation:** Validates types, non-negative values, humidity boundaries ($0$–$100\%$), and ensures $T_{\text{min}} \le T \le T_{\text{max}}$.
- **Intelligent Warnings:** Automatically flags date queries falling outside training/dataset boundaries and alerts on historical holiday mismatches.
- **Auditing:** Every prediction is saved to MongoDB before the HTTP response is returned.

---

## 14. MongoDB Integration

- **Database:** `smart_mobility`
- **Collection:** `prediction_history`
- **Driver:** `pymongo` (with reusable client and ping connectivity verification)
- **Document Structure:** Stores the complete input payload, model forecast, baseline metrics, capacity calculation, station distribution, and UTC timestamp ending in `Z` (`YYYY-MM-DDTHH:MM:SS.mmmZ`).
- **Response Fields:** Includes `prediction_record_id` (string ObjectId) and `created_at` timestamp.

---

## 15. API Endpoint Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status, project title, and route overview |
| `GET` | `/health` | Health check verifying model load state and MongoDB connection |
| `POST` | `/predict` | Full prediction pipeline, capacity sizing, and MongoDB storage |
| `GET` | `/history` | Paginated prediction audit records (`limit`, `offset`) |
| `GET` | `/history/{record_id}` | Retrieve a specific prediction record by MongoDB ObjectId |
| `GET` | `/docs` | Interactive Swagger UI documentation |

---

## 16. Postman Testing Instructions

### Sample POST `/predict` Request

- **Method:** `POST`
- **URL:** `http://127.0.0.1:8000/predict`
- **Headers:** `Content-Type: application/json`
- **Body:**
```json
{
  "date": "2023-11-07",
  "hour": 8,
  "temperature": 4.5,
  "max_temperature": 7.0,
  "min_temperature": 2.0,
  "wind_speed": 4.2,
  "precipitation": 0.0,
  "humidity": 80.0,
  "weather": 1,
  "sunshine": 30.0,
  "season": 4,
  "is_holiday": false
}
```

### Sample Response
```json
{
  "input": { ... },
  "prediction": {
    "predicted_hourly_bicycle_trips": 41.04,
    "demand_profile": "Contextual demand profile 1",
    "profile_average_demand": 20.57,
    "profile_statistics": { ... }
  },
  "baseline": {
    "weekday_hour_baseline_demand": 35.48,
    "difference_from_baseline_pct": 15.67
  },
  "capacity": {
    "recommendation": "Increase bicycle availability",
    "estimated_concurrent_bicycles": 10.04,
    "recommended_concurrent_capacity": 12.05,
    "mean_trip_duration_minutes": 14.68,
    "safety_buffer_pct": 20
  },
  "station_allocation": {
    "day_of_week": 1,
    "hour": 8,
    "total_allocated_stations": 8,
    "stations": [ ... ]
  },
  "warnings": [
    "The requested date/hour belongs to the held-out evaluation period."
  ],
  "prediction_record_id": "6b011cb626066ae1b38a6f41",
  "created_at": "2026-11-21T04:23:18.231Z"
}
```

---

## 17. Local Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/NExind/Smart-Mobility-Demand-Forecasting.git
cd Smart-Mobility-Demand-Forecasting
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy `.env.example` to `.env` (adjust MongoDB URI if not using local default):
```bash
cp .env.example .env
```

### 5. Start MongoDB
Ensure MongoDB is running locally on port `27017` (e.g., via MongoDB Community Server or Compass).

### 6. Run the FastAPI Application
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.

---

## 18. Public Deployment with ngrok

To expose your local FastAPI server to remote clients or Postman via a secure public HTTPS URL:

1. Install ngrok and authenticate:
   ```bash
   ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
   ```
2. In a second terminal, start the tunnel on port 8000:
   ```bash
   ngrok http 8000
   ```
3. Copy the public forwarding HTTPS URL (e.g. `https://xxxx.ngrok-free.app`).
4. Test the remote endpoints in Postman (`https://xxxx.ngrok-free.app/predict`).

> [!WARNING]
> Only expose port `8000` (FastAPI). Never tunnel MongoDB port `27017` or expose database credentials.

---

## 19. Project Limitations

- **Single-City Scope:** Fitted on Bergen municipal mobility data; geographic transferability requires retraining with local station graphs and weather profiles.
- **Top-Down Station Allocation:** Predictions are generated city-wide and distributed via historical proportions, rather than through individual graph-neural-network or station-specific regressions.
- **Calendar Boundary:** Evaluated on 2023 observations; queries outside 2023 represent extrapolations.
- **Micro-Mobility Exclusion:** Electric scooters and shared mopeds are excluded from demand estimates.
- **Static Fleet Capacity:** Concurrency formulas use average trip duration and safety buffers, rather than real-time GPS telemetry or dock occupancy sensors.

---

## 20. Running the Jupyter Notebook

The canonical training pipeline is preserved in [`bike_demand_training.ipynb`](bike_demand_training.ipynb).

### Execution Steps
1. Place `a.csv` in `data/a.csv`.
2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook bike_demand_training.ipynb
   ```
3. Run all cells in sequence (`Kernel -> Restart and Run All`).
4. The notebook will:
   - Perform trip filtering and time-series aggregation.
   - Plot hourly demand distributions and temporal patterns.
   - Fit K-Means clustering and display cluster statistics.
   - Train HistGradientBoostingRegressor and evaluate on the test split.
   - Output comparison plots (Actual vs. Model vs. Baseline).
   - Export the serialized artifact to `artifacts/bike_demand_model.pkl`.

---

## 21. Documentation & Model Artifacts

- **Model Card:** [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) (19-section academic model card)
- **Inference Guide:** [`docs/INFERENCE_GUIDE.md`](docs/INFERENCE_GUIDE.md) (standalone Python inference)
- **Architecture Overview:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Dataset Guide:** [`data/README.md`](data/README.md)
- **Model Artifacts Guide:** [`artifacts/README.md`](artifacts/README.md)
- **Kaggle Publishing Package:** [`kaggle_export/`](kaggle_export/)
