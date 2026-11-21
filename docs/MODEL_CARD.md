# Model Card: Smart Mobility Demand Forecasting and Capacity Allocation

## 1. Model Overview

* **Model Name:** Smart Mobility Hourly Bicycle Demand Predictor & Fleet Capacity Allocator
* **Model Version:** 2.0.0 (Artifact Version 2)
* **Framework:** Scikit-learn (`Pipeline`, `HistGradientBoostingRegressor`, `KMeans`, `ColumnTransformer`)
* **Task:** Hourly time-series regression and contextual profile clustering for micro-mobility fleet operations
* **Target Variable:** Total shared-bicycle departures (`classic_bike` + `electric_bike`) within one city-wide hour
* **Primary Maintainers:** Academic Project Development Team

This model package combines unsupervised contextual clustering (K-Means) with non-linear gradient-boosted decision trees (`HistGradientBoostingRegressor`) to forecast shared-bicycle departures. The package also bundles a historical weekday-hour baseline, concurrency-based bicycle capacity recommendation logic, and historical station pickup distribution shares.

---

## 2. Problem Statement

Urban shared micro-mobility systems face dynamic imbalances in vehicle supply and demand driven by time-of-day, day-of-week, seasonality, and volatile weather conditions. Inadequate fleet availability during peak hours leads to unfulfilled travel demand, while excess supply during low-demand periods incurs idle capital depreciation and station clutter. 

The objective is to:
1. Accurately predict city-wide hourly bicycle demand from temporal, meteorological, and calendar context.
2. Segment operating periods into interpretable contextual demand profiles using unsupervised clustering.
3. Compare predictions against a robust historical weekday-hour baseline to identify structural anomalies.
4. Estimate concurrent active bicycle fleet requirements and recommend whether to increase, maintain, or reduce availability.
5. Allocate predicted city-wide demand across stations using historical pickup patterns.

---

## 3. Intended Use

* **Primary Use Case:** Operational decision support for shared bicycle operators (fleet managers, rebalancing teams, and dispatchers).
* **Target Users:** Transportation planners, micro-mobility operations engineers, and academic researchers evaluating machine learning applications in smart cities.
* **Out-of-Scope Uses:**
  * Not intended for electric scooter demand prediction (scooters were explicitly excluded from the demand target).
  * Not intended for individual station-level machine learning forecasting (station allocation is proportional/historical, not a distinct station-level ML model).
  * Not suitable for direct deployment in cities other than Bergen, Norway without retraining on local mobility and meteorological records.

---

## 4. Model Architecture

The model architecture is a composite, multi-stage pipeline:

```
Raw Input Context (12 fields: date, hour, weather, season, holiday)
  │
  ├── Feature Engineering: Calendar attributes & cyclic sine/cosine encodings
  │
  ├── Stage 1: Contextual Clustering (Unsupervised)
  │     StandardScaler + OneHotEncoder → KMeans (4 clusters)
  │     Output: Contextual demand profile category (Profile 1–4)
  │
  ├── Stage 2: Hourly Demand Regression (Supervised)
  │     ColumnTransformer (numeric passthrough + OneHotEncoder for categoricals & profile)
  │     → HistGradientBoostingRegressor (loss='squared_error', max_iter=300)
  │     → Output: Raw prediction clipped to lower bound of 0.0
  │
  ├── Stage 3: Baseline Comparison & Anomaly Check
  │     Historical mean lookup: (day_of_week, hour)
  │     Percentage deviation calculation
  │
  ├── Stage 4: Fleet Capacity Estimation
  │     Concurrency = Predicted Demand × (Mean Trip Duration: 14.68 min) / 60
  │     Recommended Capacity = Concurrency × 1.20 (20% Safety Buffer)
  │     Recommendation: Increase (≥ +15%), Reduce (≤ -15%), Maintain (within ±15%)
  │
  └── Stage 5: Station Allocation
        Historical pickup proportion lookup for top-8 stations + remaining share
```

All preprocessing transformers, clustering estimators, regressor pipelines, and lookup tables are bundled into a single serialized Joblib artifact (`bike_demand_model.pkl`).

---

## 5. Training Dataset

* **Source File:** `a.csv` (Bergen, Norway shared mobility network)
* **Temporal Coverage:** January 1, 2023, 04:14:20 UTC to December 31, 2023, 22:35:40 UTC
* **Total Trip Records:** 467,861 trips
  * Electric Bike: 200,015 trips (42.75%)
  * Electric Scooter: 198,809 trips (42.49%) — **Excluded from demand target**
  * Classic Bike: 69,037 trips (14.76%)
* **Target Trips Included:** 269,052 bicycle trips (`classic_bike` + `electric_bike`)
* **Aggregation Granularity:** Hourly intervals across the continuous 2023 timeline (8,755 hours total).
* **Zero-Demand Handling:** 1,882 quiet hours with zero bicycle departures were preserved with target value `0` to prevent selection bias.
* **Missing Value Imputation:**
  * Numerical weather gaps: Hourly trip medians $\rightarrow$ Daily same-date medians fallback $\rightarrow$ Time-based linear interpolation $\rightarrow$ Global training median.
  * Weather codes and season: Modal observations per hour $\rightarrow$ Modal observation per date $\rightarrow$ Forward/backward fill.

---

## 6. Target Variable

* **Variable Name:** `bike_demand` (`hourly_bicycle_trip_starts`)
* **Definition:** Count of distinct `classic_bike` and `electric_bike` trip departures initiated within each 60-minute window across the municipal service area.
* **Target Statistics (Training Period: 7,004 hours):**
  * Mean: 32.98 trips/hour
  * Median: 21.0 trips/hour
  * Standard Deviation: 34.62 trips/hour
  * Minimum: 0 trips/hour
  * Maximum: 206 trips/hour

---

## 7. Feature Engineering

The model accepts **12 raw input fields** and deterministically computes **8 derived temporal features**:

### User-Supplied Fields (12)
1. `date`: `YYYY-MM-DD` string
2. `hour`: Integer `[0, 23]`
3. `temperature`: Hourly air temperature (°C)
4. `max_temperature`: Daily maximum temperature (°C)
5. `min_temperature`: Daily minimum temperature (°C)
6. `wind_speed`: Mean wind speed (m/s), $\ge 0$
7. `precipitation`: Hourly precipitation accumulation (mm), $\ge 0$
8. `humidity`: Relative humidity (0–100%)
9. `weather`: Categorical integer weather code
10. `sunshine`: Hourly sunshine duration (minutes), $\ge 0$
11. `season`: Categorical season code (`0`: Winter, `1`: Spring, `2`: Summer, `3`: Autumn)
12. `is_holiday`: Boolean public holiday indicator

### Derived Temporal & Cyclic Features (8)
* `day_of_week`: Integer `[0, 6]` (Monday=0, Sunday=6)
* `is_weekend`: Binary flag (`day_of_week >= 5`)
* `month`: Integer `[1, 12]`
* `day_of_year`: Integer `[1, 366]`
* Cyclic sine/cosine pairs to capture continuous periodic boundary continuity (e.g., 23:00 wrapping to 00:00, December wrapping to January):
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
  $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
  $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot (\text{month}-1)}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot (\text{month}-1)}{12}\right)$$
  $$\text{doy\_sin} = \sin\left(\frac{2\pi \cdot (\text{doy}-1)}{365.25}\right), \quad \text{doy\_cos} = \cos\left(\frac{2\pi \cdot (\text{doy}-1)}{365.25}\right)$$

---

## 8. K-Means Clustering for Contextual Profiles

* **Algorithm:** `sklearn.cluster.KMeans` ($k=4$, `random_state=42`, `n_init=20`)
* **Input Features:** 17 normalized numerical features (weather + cyclic calendar) + 2 one-hot encoded categorical features (`weather_code`, `season`). **Target demand is strictly excluded from clustering inputs.**
* **Discovered Cluster Distribution (Training Data):**

| Cluster Profile | Hours | % Training | Mean Demand | Median | Std | P10 | P25 | P75 | P90 | Primary Environmental Characteristics |
|---|---|---|---|---|---|---|---|---|---|---|
| **Contextual profile 1** | 1,937 | 27.7% | 20.57 | 14.0 | 22.93 | 0.0 | 2.0 | 30.0 | 54.0 | Cold weather, winter/early spring, nighttime hours |
| **Contextual profile 2** | 2,094 | 29.9% | 31.76 | 21.0 | 34.35 | 0.0 | 3.0 | 47.0 | 78.0 | Warm, humid, late-summer/autumn daytime |
| **Contextual profile 3** | 1,549 | 22.1% | 31.92 | 24.0 | 33.41 | 0.0 | 3.0 | 48.0 | 78.2 | Cool, dry, spring daytime conditions |
| **Contextual profile 4** | 1,424 | 20.3% | 52.80 | 50.0 | 43.85 | 0.0 | 10.8 | 79.0 | 108.0 | Mild, sunny commute peak windows |

* **Important Note on Cluster Labels:** Statistical analysis revealed that Profile 2 and Profile 3 have nearly identical mean demand (31.76 vs 31.92 trips/hour, Cohen's $d = 0.004$), yet represent distinct meteorological regimes (warm/humid vs cool/dry). Hence, neutral contextual labels (`Contextual demand profile 1–4`) are used rather than artificial demand rankings. **The profile describes contextual similarity, whereas the regression model predicts numerical demand.**

---

## 9. HistGradientBoosting Regression

* **Algorithm:** `sklearn.ensemble.HistGradientBoostingRegressor`
* **Hyperparameters:**
  * `loss`: `squared_error`
  * `learning_rate`: `0.06`
  * `max_iter`: `300`
  * `max_leaf_nodes`: `20`
  * `min_samples_leaf`: `25`
  * `l2_regularization`: `2.0`
  * `early_stopping`: `True` (`n_iter_no_change=10`, `validation_fraction=0.1`)
  * `random_state`: `42`
* **Post-Processing:** Predictions are non-negatively clipped: $\hat{y} = \max(0.0, \hat{y}_{\text{raw}})$.

---

## 10. Model Training Methodology

* Fitted strictly on the training partition.
* Preprocessing transformers (`StandardScaler`, `OneHotEncoder`) were fit on training data only and applied via pipelines to prevent data leakage.
* Regression features incorporate all environmental features plus the assigned contextual demand profile from K-Means.

---

## 11. Chronological Train/Test Split

* **Strict Temporal Ordering:** To replicate genuine production forecasting, the timeline was split chronologically without random shuffling:
  * **Training Period:** January 1, 2023, 04:00:00 to October 19, 2023, 23:00:00 (7,004 hourly periods, 80.0%)
  * **Held-out Test Period:** October 20, 2023, 00:00:00 to December 31, 2023, 22:00:00 (1,751 hourly periods, 20.0%)
* This split evaluates the model on later, unseen autumn and winter seasons.

---

## 12. Evaluation Metrics

Evaluated on the 1,751 held-out chronological test observations:

| Metric | ML Model (HistGradientBoosting + KMeans) | Weekday-Hour Historical Baseline | Error Reduction / Relative Gain |
|---|---|---|---|
| **MAE (trips/hour)** | **7.225** | 15.050 | **51.99% lower error** |
| **RMSE (trips/hour)** | **11.166** | 22.710 | **50.83% lower error** |
| **$R^2$ Score** | **0.818** | 0.246 | **+0.572 variance explained** |

The ML model halves the prediction error compared to the historical baseline and captures over 81.7% of the demand variance during the held-out period.

---

## 13. Baseline Comparison

The simple reference baseline computes the historical average training demand for each `(day_of_week, hour)` tuple (168 bins). If an unobserved slot occurs, it falls back to the global training mean (32.98 trips/hour).
* The baseline uses no meteorological information, isolating the incremental predictive power gained from weather and clustering context.

---

## 14. Capacity Estimation

Fleet concurrency is estimated using Little's Law approximation:
$$\text{Estimated Concurrent Bicycles} = \text{Predicted Hourly Demand} \times \frac{\bar{T}_{\text{trip}}}{60}$$
where $\bar{T}_{\text{trip}} = 14.68\text{ minutes}$ (mean bicycle trip duration from training data).

* **Safety Buffer:** A 20% planning buffer is applied:
  $$\text{Recommended Capacity} = \text{Estimated Concurrent Bicycles} \times 1.20$$
* **Operational Recommendation Rule:**
  * $\frac{\hat{y} - y_{\text{baseline}}}{y_{\text{baseline}}} \ge +15\% \implies$ **"Increase bicycle availability"**
  * $\frac{\hat{y} - y_{\text{baseline}}}{y_{\text{baseline}}} \le -15\% \implies$ **"Reduce excess bicycle availability"**
  * Otherwise $\implies$ **"Maintain current bicycle availability"**

---

## 15. Station Allocation

* **Methodology:** Predicted city-wide trips are distributed to individual stations using historical departure proportions conditioned on `(day_of_week, hour)`.
* **Top Stations Tracked:** The top 8 historical stations for each time slot are reported with expected pickup counts, while remaining demand is aggregated under `remaining_stations`.
* **Important Distinction:** This is a share-based empirical allocation, **not** an independent station-level machine learning model.

---

## 16. Limitations

1. **Geographic Specificity:** Built on 2023 mobility and weather data for Bergen, Norway. Infrastructure, topography, and commuter habits differ across cities.
2. **Calendar Coverage:** Trained on 2023. Dates outside 2023 represent extrapolations where multi-year trend shifts and inflation are not captured.
3. **Scooter Exclusion:** Does not forecast electric scooter demand (198,809 trips were filtered out).
4. **Extreme Weather Shocks:** Severe storms or unobserved weather codes outside the training range will produce less reliable estimates.
5. **No Real-Time Telemetry:** The model forecasts demand from weather and calendar inputs; it does not consume real-time station inventory, GPS positions, or maintenance lockouts.

---

## 17. Installation Requirements

```bash
pip install scikit-learn>=1.4.0 pandas>=2.0.0 numpy>=1.24.0 joblib>=1.3.0
```

---

## 18. Example Inference

```python
import joblib
import pandas as pd
import numpy as np
import math

# 1. Load the model bundle
bundle = joblib.load("bike_demand_model.pkl")

# 2. Input context (e.g., commute peak hour)
request = {
    "date": "2023-05-15",
    "hour": 8,
    "temperature": 14.2,
    "max_temperature": 17.0,
    "min_temperature": 10.5,
    "wind_speed": 2.8,
    "precipitation": 0.0,
    "humidity": 68.0,
    "weather": 3,
    "sunshine": 55.0,
    "season": 1,
    "is_holiday": False
}

# 3. Derive calendar & cyclic features
ts = pd.Timestamp(request["date"]) + pd.Timedelta(hours=request["hour"])
dow = int(ts.dayofweek)
doy = int(ts.dayofyear)
month = int(ts.month)

row = {
    **{k: float(request[k]) for k in ["temperature", "max_temperature", "min_temperature",
                                      "wind_speed", "precipitation", "humidity", "sunshine"]},
    "hour_sin": math.sin(2 * math.pi * request["hour"] / 24),
    "hour_cos": math.cos(2 * math.pi * request["hour"] / 24),
    "dow_sin": math.sin(2 * math.pi * dow / 7),
    "dow_cos": math.cos(2 * math.pi * dow / 7),
    "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
    "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
    "day_of_year_sin": math.sin(2 * math.pi * (doy - 1) / 365.25),
    "day_of_year_cos": math.cos(2 * math.pi * (doy - 1) / 365.25),
    "is_holiday": int(request["is_holiday"]),
    "is_weekend": int(dow >= 5),
    "weather_code": str(int(request["weather"])),
    "season": str(int(request["season"])),
}

# 4. Predict contextual profile
ctx_df = pd.DataFrame([row])
ctx_cols = bundle["cluster_numeric_features"] + bundle["cluster_categorical_features"]
cluster_mat = bundle["cluster_preprocessor"].transform(ctx_df[ctx_cols])
cluster_id = int(bundle["kmeans"].predict(cluster_mat)[0])
profile_name = bundle["cluster_label_map"][cluster_id]
row["demand_profile"] = profile_name

# 5. Predict hourly demand
feat_df = pd.DataFrame([row])
feat_cols = bundle["model_numeric_features"] + bundle["model_categorical_features"]
prediction = max(0.0, float(bundle["model"].predict(feat_df[feat_cols])[0]))

# 6. Capacity recommendation
concurrent = prediction * bundle["mean_trip_duration_minutes"] / 60.0
capacity = concurrent * bundle["safety_buffer"]

print(f"Predicted Demand: {prediction:.1f} trips/hour")
print(f"Assigned Profile: {profile_name}")
print(f"Estimated Concurrent Bicycles: {concurrent:.1f}")
print(f"Recommended Capacity (with buffer): {capacity:.1f}")
```

---

## 19. Dataset Provenance and Licensing Information

* **Dataset Origin:** The underlying dataset (`a.csv`) contains trip trajectories and station coordinate records corresponding to the municipal shared micro-mobility system of Bergen, Norway (operating under the Bergen Bysykkel network) matched with regional meteorological observation stations for 2023.
* **Licensing Status:** The original trip-level dataset was supplied as an internal academic project asset without an attached third-party license file. Because municipal open mobility feeds are generally published under open data frameworks (such as NLOD or CC BY 4.0) but upstream redistribution terms were not explicitly bundled, **licensing uncertainty exists**.
* **Model Licensing:** This trained model artifact, documentation, and inference scripts are released under **CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)** for educational, research, and non-commercial evaluation purposes. Users seeking commercial deployment must independently verify licensing terms with the underlying micro-mobility operator and meteorological provider.
