# System Architecture & Technical Specifications

**Project:** Smart Mobility Demand Forecasting and Capacity Allocation  
**Implementation:** API-First Backend (FastAPI + Scikit-Learn + MongoDB)

---

## 1. High-Level System Overview

```
                ┌───────────────────────────────────────────────┐
                │          Client (Postman / ngrok)             │
                └───────────────────────┬───────────────────────┘
                                        │ HTTP POST /predict (JSON)
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Application (main.py)                                                 │
│                                                                               │
│  1. Pydantic Request Validation                                               │
│     - 12 fields (date, hour, weather, temp, sunshine, holiday, season)        │
│     - Range & consistency checks (min_temp <= temp <= max_temp)               │
│                                                                               │
│  2. Feature Engineering                                                       │
│     - Cyclic Sin/Cos encodings for hour, weekday, month, day of year          │
│     - Weekend and holiday indicators                                          │
│                                                                               │
│  3. Warning Classification                                                    │
│     - Training vs. evaluation vs. extrapolation date boundaries                │
│     - Historical holiday consistency checks                                   │
│                                                                               │
│  4. Two-Stage ML Inference                                                    │
│     ┌───────────────────────────────────────────────────────────────┐         │
│     │ Stage 1: Contextual Clustering (K-Means, 4 Profiles)          │         │
│     │ StandardScaler + OneHotEncoder -> K-Means -> demand_profile   │         │
│     └───────────────────────────────┬───────────────────────────────┘         │
│                                     │                                         │
│     ┌───────────────────────────────▼───────────────────────────────┐         │
│     │ Stage 2: Hourly Demand Regression                             │         │
│     │ ColumnTransformer + HistGradientBoostingRegressor             │         │
│     │ Output: Predicted Hourly Departures                           │         │
│     └───────────────────────────────┬───────────────────────────────┘         │
│                                     │                                         │
│  5. Operational Baseline & Capacity Allocation                                │
│     - Weekday-Hour Baseline lookup & % difference                             │
│     - Concurrency estimation (turnover = 60 / avg_duration)                   │
│     - Safety buffer application (recommended capacity)                        │
│     - Top-station pickup share distribution                                   │
│                                                                               │
│  6. Database Persistence                                                      │
│     - Reusable PyMongo client                                                 │
│     - BSON conversion with UTC timestamp ('Z' suffix)                         │
└───────────────────────┬───────────────────────────────────────────────┬───────┘
                        │                                               │
                        ▼                                               ▼
         ┌───────────────────────────────┐               ┌───────────────────────────────┐
         │ Production Artifact (.pkl)    │               │ MongoDB Persistence           │
         │ - Fitted KMeans & Pipeline    │               │ - smart_mobility db           │
         │ - Baseline lookup (168 bins)  │               │ - prediction_history coll     │
         │ - Station shares (146 bins)   │               │                               │
         └───────────────────────────────┘               └───────────────────────────────┘
```

---

## 2. Machine Learning Architecture

### Stage 1: Unsupervised Contextual Clustering (K-Means)
- **Objective:** Discover contextual weather-calendar patterns rather than assuming rigid time-of-day blocks.
- **Input Features (19):** 7 meteorological metrics (`temperature`, `max_temperature`, `min_temperature`, `wind_speed`, `precipitation`, `humidity`, `sunshine`), 8 cyclic encodings (`hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `day_of_year_sin/cos`), 2 binary flags (`is_holiday`, `is_weekend`), and 2 categorical features (`weather_code`, `season`).
- **Cluster Count:** $k=4$, initialized with $n_{\text{init}}=20$ and `random_state=42`.
- **Profiles:** Sorted by average training demand and assigned neutral labels (`Contextual demand profile 1` through `4`).

### Stage 2: Supervised Hourly Demand Regression (HistGradientBoosting)
- **Objective:** Predict total city-wide bicycle departures (classic + electric bikes).
- **Features (20):** Numerical context features + one-hot encoded categoricals (`weather_code`, `season`, and the assigned `demand_profile` from Stage 1).
- **Algorithm:** `HistGradientBoostingRegressor` with:
  - `learning_rate = 0.06`
  - `max_iter = 300`
  - `max_leaf_nodes = 20`
  - `min_samples_leaf = 25`
  - `l2_regularization = 2.0`
  - `early_stopping = True`

---

## 3. Capacity & Station Allocation Methodology

### Fleet Sizing & Concurrency
Demand is converted from total hourly trip starts to concurrent bicycles in active service:

$$\text{Turnover Rate} = \frac{60}{\bar{d}} \quad (\text{trips per bicycle per hour})$$

$$\text{Estimated Concurrent Bicycles} = \frac{\hat{Y}}{\text{Turnover Rate}} = \frac{\hat{Y} \times \bar{d}}{60}$$

$$\text{Recommended Capacity} = \left\lceil \text{Estimated Concurrent Bicycles} \times (1 + \beta) \right\rceil$$

Where:
- $\hat{Y}$: Predicted hourly departures.
- $\bar{d}$: Mean bicycle trip duration ($14.68\text{ minutes}$ during training).
- $\beta$: Operational safety buffer ($0.20$, or $20\%$).

### Station Pickup Distribution
Station allocation is computed by projecting predicted city-wide demand $\hat{Y}$ onto the historical station pickup share distribution for that specific `(day_of_week, hour)`:

$$\hat{y}_{s} = \hat{Y} \times \omega_{s, \text{dow}, \text{hour}}$$

Where $\omega_{s, \text{dow}, \text{hour}}$ is the empirical proportion of departures originating at station $s$.

---

## 4. Database Schema (MongoDB)

Predictions are stored in collection `prediction_history` within database `smart_mobility`:

```json
{
  "_id": ObjectId("6b011cb626066ae1b38a6f41"),
  "input": {
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
  },
  "prediction": {
    "predicted_hourly_bicycle_trips": 41.04,
    "demand_profile": "Contextual demand profile 1",
    "profile_average_demand": 20.57,
    "profile_description": "...",
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
  "training_period": {
    "start": "2023-01-01 04:00:00",
    "end": "2023-10-19 23:00:00"
  },
  "warnings": [],
  "created_at": ISODate("2026-11-21T04:23:18.231Z")
}
```
