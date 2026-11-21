# Dataset Directory

## Overview

This directory is designated for the shared-mobility trip dataset required by the training pipeline (`train_model.py` and `bike_demand_training.ipynb`).

The primary data file expected here is:
```
data/a.csv
```

---

## Git Exclusion Notice

> [!IMPORTANT]
> The raw data file `data/a.csv` is deliberately excluded from version control via `.gitignore` for two reasons:
> 1. **File Size:** The raw CSV is **102.7 MB** (467,861 rows), which exceeds GitHub's standard 100 MB per-file limit and requires Git LFS.
> 2. **Licensing & Redistribution:** The raw dataset originates from municipal open-data feeds (Bergen Bysykkel) covering urban trip records with station names and timestamps. To comply with redistribution terms and avoid hosting large proprietary or third-party datasets in the code repository, raw trip data is kept locally.

---

## Dataset Schema

When `data/a.csv` is provided, it contains **467,861 raw trip records** spanning January 1, 2023 to December 31, 2023 with 29 columns:

| Column Group | Columns | Description |
|---|---|---|
| **Timestamps** | `start_time`, `end_time`, `start_clock_time`, `end_clock_time` | Trip start and end ISO timestamps and clock times |
| **Trip Metrics** | `duration`, `vehicle_type` | Trip duration in seconds; vehicle modality |
| **Origin Station** | `start_station_id`, `start_station_name`, `start_station_latitude`, `start_station_longitude` | Station identifiers and geographic coordinates |
| **Destination Station** | `end_station_id`, `end_station_name`, `end_station_latitude`, `end_station_longitude` | Station identifiers and geographic coordinates |
| **Weather Observations** | `temperature`, `max_temperature`, `min_temperature`, `wind_speed`, `precipitation`, `humidity`, `sunshine`, `weather` | Hourly and daily meteorological measurements |
| **Calendar & Temporal** | `season`, `is_holiday`, `is_weekend`, `date`, `hour`, `day_of_week`, `month` | Calendar features and public holiday flags |

---

## Vehicle Types & Target Definition

The raw data contains three vehicle categories:
- **`classic_bike`:** 69,037 trips
- **`electric_bike`:** 200,015 trips
- **`electric_scooter`:** 198,809 trips

### Target Rule:
This project forecasts **shared bicycle demand**. Therefore:
- `classic_bike` and `electric_bike` trips are combined into the hourly departure target ($269,052$ total bicycle trips).
- `electric_scooter` trips are strictly excluded from the demand target.

---

## How to Set Up the Dataset Locally

1. Place the uncompressed `a.csv` file directly inside the `data/` directory:
   ```
   Smart-Mobility-Demand-Forecasting/
   └── data/
       ├── README.md
       └── a.csv
   ```
2. Verify the file path and size:
   ```bash
   python -c "from pathlib import Path; p = Path('data/a.csv'); print('Found:', p.exists(), 'Size:', p.stat().st_size if p.exists() else 0)"
   ```
3. Run model training:
   ```bash
   python train_model.py
   ```
   Or execute `bike_demand_training.ipynb` in Jupyter Notebook / VS Code.
