# Smart Mobility Model — Inference Guide

This guide explains how to load and use the `bike_demand_model.pkl` artifact in any standalone Python environment (including local scripts, Kaggle Notebooks, cloud functions, or batch pipelines) without requiring FastAPI or MongoDB.

---

## 1. Environment & Dependencies

Install the core dependencies:

```bash
pip install scikit-learn>=1.4.0 pandas>=2.0.0 numpy>=1.24.0 joblib>=1.3.0
```

---

## 2. Artifact Architecture

The artifact `bike_demand_model.pkl` is a single serialized Python dictionary (`joblib.dump`) containing:

| Key | Object Type | Purpose |
|---|---|---|
| `model` | `sklearn.pipeline.Pipeline` | Fitted HistGradientBoosting regression pipeline |
| `cluster_preprocessor` | `ColumnTransformer` | Scaler + OneHotEncoder for clustering inputs |
| `kmeans` | `sklearn.cluster.KMeans` | 4-cluster model for contextual profile assignment |
| `cluster_label_map` | `dict` | Maps numeric cluster IDs to descriptive profile strings |
| `cluster_demand_means` | `dict` | Training-period mean demand per cluster |
| `cluster_detail_stats` | `dict` | Detailed stats (p10, p25, median, p75, p90, etc.) per profile |
| `baseline_by_dow_hour` | `dict` | Historical mean bicycle demand by `(day_of_week, hour)` |
| `overall_training_mean` | `float` | Fallback mean demand across the entire training period (32.98) |
| `mean_trip_duration_minutes` | `float` | Mean duration of bicycle trips during training (14.68 min) |
| `safety_buffer` | `float` | Operational buffer multiplier (1.20 = 20% safety buffer) |
| `station_share_lookup` | `dict` | Top 8 station pickup shares conditioned on `(day_of_week, hour)` |
| `global_station_shares` | `list` | Global fallback station pickup shares |
| `model_numeric_features` | `list` | Feature order for regression numeric columns |
| `model_categorical_features` | `list` | Feature order for regression categorical columns |
| `cluster_numeric_features` | `list` | Feature order for clustering numeric columns |
| `cluster_categorical_features` | `list` | Feature order for clustering categorical columns |
| `holiday_lookup` | `dict` | Ground truth holiday flags for 2023 dates |
| `known_weather_codes` | `list` | Validated weather codes observed in training data |
| `known_season_codes` | `list` | Validated season codes observed in training data |

---

## 3. Step-by-Step Inference Process

### Step 1: Prepare the 12 User Inputs
The model expects 12 raw features:

```python
sample_input = {
    "date": "2023-05-15",          # YYYY-MM-DD
    "hour": 8,                     # 0 to 23
    "temperature": 14.2,           # Celsius
    "max_temperature": 17.0,       # Celsius (must be >= temperature)
    "min_temperature": 10.5,       # Celsius (must be <= temperature)
    "wind_speed": 2.8,             # m/s (>= 0)
    "precipitation": 0.0,          # mm (>= 0)
    "humidity": 68.0,              # percent [0, 100]
    "weather": 3,                  # integer weather code
    "sunshine": 55.0,              # minutes (>= 0)
    "season": 1,                   # integer season code (0: Winter, 1: Spring, 2: Summer, 3: Autumn)
    "is_holiday": False            # boolean
}
```

### Step 2: Feature Engineering
Derive calendar and cyclic time signals:

```python
import math
import pandas as pd

timestamp = pd.Timestamp(sample_input["date"]) + pd.Timedelta(hours=sample_input["hour"])
dow = int(timestamp.dayofweek)
doy = int(timestamp.dayofyear)
month = int(timestamp.month)

feature_row = {
    "temperature": float(sample_input["temperature"]),
    "max_temperature": float(sample_input["max_temperature"]),
    "min_temperature": float(sample_input["min_temperature"]),
    "wind_speed": float(sample_input["wind_speed"]),
    "precipitation": float(sample_input["precipitation"]),
    "humidity": float(sample_input["humidity"]),
    "sunshine": float(sample_input["sunshine"]),
    "hour_sin": math.sin(2 * math.pi * sample_input["hour"] / 24),
    "hour_cos": math.cos(2 * math.pi * sample_input["hour"] / 24),
    "dow_sin": math.sin(2 * math.pi * dow / 7),
    "dow_cos": math.cos(2 * math.pi * dow / 7),
    "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
    "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
    "day_of_year_sin": math.sin(2 * math.pi * (doy - 1) / 365.25),
    "day_of_year_cos": math.cos(2 * math.pi * (doy - 1) / 365.25),
    "is_holiday": int(sample_input["is_holiday"]),
    "is_weekend": int(dow >= 5),
    "weather_code": str(int(sample_input["weather"])),
    "season": str(int(sample_input["season"])),
}
```

### Step 3: Assign Contextual Demand Profile
Pass the environmental context to the K-Means pipeline:

```python
ctx_df = pd.DataFrame([feature_row])
cluster_cols = bundle["cluster_numeric_features"] + bundle["cluster_categorical_features"]
ctx_matrix = bundle["cluster_preprocessor"].transform(ctx_df[cluster_cols])
cluster_id = int(bundle["kmeans"].predict(ctx_matrix)[0])
demand_profile = bundle["cluster_label_map"][cluster_id]

# Inject the assigned profile as a feature for the regression model
feature_row["demand_profile"] = demand_profile
```

### Step 4: Predict Hourly Bicycle Departures
Run the regression pipeline and clip negative values to zero:

```python
feat_df = pd.DataFrame([feature_row])
model_cols = bundle["model_numeric_features"] + bundle["model_categorical_features"]
raw_pred = float(bundle["model"].predict(feat_df[model_cols])[0])
predicted_demand = round(max(0.0, raw_pred), 2)
```

### Step 5: Compare Against Baseline
Look up the historical average demand for the same weekday and hour:

```python
baseline_demand = round(
    bundle["baseline_by_dow_hour"].get((dow, sample_input["hour"]), bundle["overall_training_mean"]),
    2
)
pct_diff = round(((predicted_demand - baseline_demand) / baseline_demand) * 100, 2) if baseline_demand > 0 else 0.0
```

### Step 6: Compute Capacity & Fleet Recommendations
Approximate active fleet concurrency and apply the 20% safety buffer:

```python
mean_dur = bundle["mean_trip_duration_minutes"]
concurrent_bikes = round(predicted_demand * mean_dur / 60.0, 2)
recommended_capacity = round(concurrent_bikes * bundle["safety_buffer"], 2)

if pct_diff >= 15.0:
    recommendation = "Increase bicycle availability"
elif pct_diff <= -15.0:
    recommendation = "Reduce excess bicycle availability"
else:
    recommendation = "Maintain current bicycle availability"
```

### Step 7: Allocate Demand to Stations
Distribute the city-wide total across top stations:

```python
shares = bundle["station_share_lookup"].get((dow, sample_input["hour"]), bundle["global_station_shares"])

allocated = []
total_share = 0.0
for s in shares:
    pickups = round(predicted_demand * s["share"], 2)
    allocated.append({
        "station_id": s["station_id"],
        "station_name": s["station_name"],
        "historical_share_pct": round(s["share"] * 100, 2),
        "expected_pickups": pickups
    })
    total_share += s["share"]

remaining_pickups = round(predicted_demand * max(0.0, 1.0 - total_share), 2)
```

---

## 4. Complete Executable Script

```python
import joblib
import pandas as pd
import numpy as np
import math

def run_smart_mobility_inference(artifact_path: str, req: dict):
    bundle = joblib.load(artifact_path)
    
    # 1. Feature Engineering
    ts = pd.Timestamp(req["date"]) + pd.Timedelta(hours=req["hour"])
    dow, doy, month = int(ts.dayofweek), int(ts.dayofyear), int(ts.month)
    
    row = {
        "temperature": float(req["temperature"]),
        "max_temperature": float(req["max_temperature"]),
        "min_temperature": float(req["min_temperature"]),
        "wind_speed": float(req["wind_speed"]),
        "precipitation": float(req["precipitation"]),
        "humidity": float(req["humidity"]),
        "sunshine": float(req["sunshine"]),
        "hour_sin": math.sin(2 * math.pi * req["hour"] / 24),
        "hour_cos": math.cos(2 * math.pi * req["hour"] / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
        "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
        "day_of_year_sin": math.sin(2 * math.pi * (doy - 1) / 365.25),
        "day_of_year_cos": math.cos(2 * math.pi * (doy - 1) / 365.25),
        "is_holiday": int(req["is_holiday"]),
        "is_weekend": int(dow >= 5),
        "weather_code": str(int(req["weather"])),
        "season": str(int(req["season"])),
    }
    
    # 2. Contextual Clustering
    ctx_df = pd.DataFrame([row])
    ctx_cols = bundle["cluster_numeric_features"] + bundle["cluster_categorical_features"]
    ctx_mat = bundle["cluster_preprocessor"].transform(ctx_df[ctx_cols])
    cluster_id = int(bundle["kmeans"].predict(ctx_mat)[0])
    profile = bundle["cluster_label_map"][cluster_id]
    row["demand_profile"] = profile
    
    # 3. Regression Demand Prediction
    feat_df = pd.DataFrame([row])
    feat_cols = bundle["model_numeric_features"] + bundle["model_categorical_features"]
    pred = round(max(0.0, float(bundle["model"].predict(feat_df[feat_cols])[0])), 2)
    
    # 4. Baseline Comparison
    baseline = round(bundle["baseline_by_dow_hour"].get((dow, req["hour"]), bundle["overall_training_mean"]), 2)
    pct_diff = round(((pred - baseline) / baseline) * 100, 2) if baseline > 0 else 0.0
    
    # 5. Capacity Estimation
    concurrent = round(pred * bundle["mean_trip_duration_minutes"] / 60.0, 2)
    capacity = round(concurrent * bundle["safety_buffer"], 2)
    rec = "Increase bicycle availability" if pct_diff >= 15 else ("Reduce excess bicycle availability" if pct_diff <= -15 else "Maintain current bicycle availability")
    
    # 6. Station Allocation
    shares = bundle["station_share_lookup"].get((dow, req["hour"]), bundle["global_station_shares"])
    top_stations = [
        {"station": s["station_name"], "pickups": round(pred * s["share"], 2), "share_pct": round(s["share"] * 100, 1)}
        for s in shares[:5]
    ]
    
    return {
        "timestamp": str(ts),
        "predicted_demand": pred,
        "baseline_demand": baseline,
        "pct_diff": pct_diff,
        "demand_profile": profile,
        "concurrent_bicycles": concurrent,
        "recommended_capacity": capacity,
        "recommendation": rec,
        "top_stations": top_stations
    }

if __name__ == "__main__":
    result = run_smart_mobility_inference(
        "bike_demand_model.pkl",
        {"date": "2023-05-15", "hour": 8, "temperature": 14.2, "max_temperature": 17.0,
         "min_temperature": 10.5, "wind_speed": 2.8, "precipitation": 0.0, "humidity": 68.0,
         "weather": 3, "sunshine": 55.0, "season": 1, "is_holiday": False}
    )
    print("Inference Result:", result)
```
