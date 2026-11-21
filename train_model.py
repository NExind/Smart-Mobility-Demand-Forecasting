from pathlib import Path
import math
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
RANDOM_STATE = 42
N_DEMAND_PROFILES = 4
TEST_FRACTION = 0.2
SAFETY_BUFFER = 1.2
TOP_STATIONS_TO_REPORT = 8
ARTIFACT_PATH = Path('artifacts') / 'bike_demand_model.pkl'
DATASET_CANDIDATES = [Path('data/a.csv'), Path('a.csv'), Path('../data/a.csv'), Path('../a.csv'), Path('data/bike_trips.csv'), Path('../data/bike_trips.csv')]
DATASET_PATH = next((p for p in DATASET_CANDIDATES if p.exists()), None)
if DATASET_PATH is None:
    raise FileNotFoundError('Could not find the dataset.  Put the supplied CSV at data/a.csv (recommended) or a.csv in the project root, then rerun.')

def main() -> None:
    print('=' * 70)
    print('Smart Mobility Demand Forecasting — Training Pipeline')
    print('=' * 70)
    print(f'\nDataset path : {DATASET_PATH.resolve()}')
    print(f'Artifact dest: {ARTIFACT_PATH.resolve()}\n')
    REQUIRED_COLUMNS = ['start_time', 'duration', 'vehicle_type', 'start_station_id', 'start_station_name', 'temperature', 'max_temperature', 'min_temperature', 'wind_speed', 'precipitation', 'humidity', 'weather', 'sunshine', 'season', 'is_holiday']
    raw = pd.read_csv(DATASET_PATH, usecols=REQUIRED_COLUMNS, parse_dates=['start_time'], low_memory=False)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f'Dataset is missing required columns: {missing_columns}')
    raw = raw.dropna(subset=['start_time', 'vehicle_type']).copy()
    raw['hour_start'] = raw['start_time'].dt.floor('h')
    raw['service_date'] = raw['start_time'].dt.normalize()
    print(f'Dataset records loaded   : {len(raw):,}')
    print(f"Trip period              : {raw['start_time'].min()} → {raw['start_time'].max()}")
    vehicle_counts = raw['vehicle_type'].value_counts()
    for vtype, count in vehicle_counts.items():
        print(f'  {vtype:20s}: {count:>8,} trips')
    BIKE_TYPES = {'classic_bike', 'electric_bike'}
    WEATHER_NUMERIC = ['temperature', 'max_temperature', 'min_temperature', 'wind_speed', 'precipitation', 'humidity', 'sunshine']
    bike_rows = raw[raw['vehicle_type'].isin(BIKE_TYPES)].copy()
    if bike_rows.empty:
        raise ValueError('No classic_bike or electric_bike rows found.')
    bicycle_trip_count = len(bike_rows)
    print(f'\nBicycle trips (target)   : {bicycle_trip_count:,}')
    print(f'Excluded scooter trips   : {len(raw) - bicycle_trip_count:,}')
    bike_demand = bike_rows.groupby('hour_start').size().rename('bike_demand')
    hourly_weather_numeric = raw.groupby('hour_start')[WEATHER_NUMERIC].median()
    hourly_weather_code = raw.groupby('hour_start')['weather'].agg(lambda s: s.mode().iloc[0])
    daily_weather_numeric = raw.groupby('service_date')[WEATHER_NUMERIC].median()
    daily_weather_code = raw.groupby('service_date')['weather'].agg(lambda s: s.mode().iloc[0])
    daily_season = raw.groupby('service_date')['season'].agg(lambda s: s.mode().iloc[0])
    daily_holiday = raw.groupby('service_date')['is_holiday'].max()
    full_hours = pd.date_range(start=raw['hour_start'].min(), end=raw['hour_start'].max(), freq='h', name='hour_start')
    hourly = pd.DataFrame(index=full_hours)
    hourly['bike_demand'] = bike_demand.reindex(full_hours, fill_value=0).astype(int)
    for column in WEATHER_NUMERIC:
        observed = hourly_weather_numeric[column].reindex(full_hours)
        daily_fallback = pd.Series([daily_weather_numeric.loc[t.normalize(), column] for t in full_hours], index=full_hours, dtype='float64')
        hourly[column] = observed.fillna(daily_fallback)
        hourly[column] = hourly[column].interpolate(method='time').ffill().bfill()
        if hourly[column].isna().any():
            hourly[column] = hourly[column].fillna(float(raw[column].median()))
    weather_observed = hourly_weather_code.reindex(full_hours)
    weather_fallback = pd.Series([daily_weather_code.loc[t.normalize()] for t in full_hours], index=full_hours)
    hourly['weather_code'] = weather_observed.fillna(weather_fallback).ffill().bfill().astype(int)
    hourly['season'] = [int(daily_season.loc[t.normalize()]) for t in full_hours]
    hourly['is_holiday'] = [int(daily_holiday.loc[t.normalize()]) for t in full_hours]
    hourly['hour'] = hourly.index.hour
    hourly['day_of_week'] = hourly.index.dayofweek
    hourly['month'] = hourly.index.month
    hourly['day_of_year'] = hourly.index.dayofyear
    hourly['is_weekend'] = (hourly['day_of_week'] >= 5).astype(int)
    hourly['hour_sin'] = np.sin(2 * np.pi * hourly['hour'] / 24)
    hourly['hour_cos'] = np.cos(2 * np.pi * hourly['hour'] / 24)
    hourly['dow_sin'] = np.sin(2 * np.pi * hourly['day_of_week'] / 7)
    hourly['dow_cos'] = np.cos(2 * np.pi * hourly['day_of_week'] / 7)
    hourly['month_sin'] = np.sin(2 * np.pi * (hourly['month'] - 1) / 12)
    hourly['month_cos'] = np.cos(2 * np.pi * (hourly['month'] - 1) / 12)
    hourly['day_of_year_sin'] = np.sin(2 * np.pi * (hourly['day_of_year'] - 1) / 365.25)
    hourly['day_of_year_cos'] = np.cos(2 * np.pi * (hourly['day_of_year'] - 1) / 365.25)
    hourly['weather_code'] = hourly['weather_code'].astype(str)
    hourly['season'] = hourly['season'].astype(str)
    assert hourly.index.is_monotonic_increasing
    assert hourly['bike_demand'].notna().all()
    assert not hourly[WEATHER_NUMERIC].isna().any().any()
    print(f'\nHourly observations      : {len(hourly):,}')
    print(f"Zero-demand hours        : {(hourly['bike_demand'] == 0).sum():,}")
    print(f"Bicycle trips aggregated : {hourly['bike_demand'].sum():,}")
    split_position = int(len(hourly) * (1 - TEST_FRACTION))
    split_timestamp = hourly.index[split_position]
    train_df = hourly.iloc[:split_position].copy()
    test_df = hourly.iloc[split_position:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError('Chronological split produced an empty set.')
    print(f'\nTraining period          : {train_df.index.min()} → {train_df.index.max()}')
    print(f'Testing period           : {test_df.index.min()} → {test_df.index.max()}')
    print(f'Training hours           : {len(train_df):,}')
    print(f'Testing hours            : {len(test_df):,}')
    CLUSTER_NUMERIC_FEATURES = ['temperature', 'max_temperature', 'min_temperature', 'wind_speed', 'precipitation', 'humidity', 'sunshine', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'is_holiday', 'is_weekend']
    CLUSTER_CATEGORICAL_FEATURES = ['weather_code', 'season']
    cluster_preprocessor = ColumnTransformer(transformers=[('numeric', StandardScaler(), CLUSTER_NUMERIC_FEATURES), ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CLUSTER_CATEGORICAL_FEATURES)], remainder='drop', sparse_threshold=0.0)
    X_train_context = train_df[CLUSTER_NUMERIC_FEATURES + CLUSTER_CATEGORICAL_FEATURES]
    X_test_context = test_df[CLUSTER_NUMERIC_FEATURES + CLUSTER_CATEGORICAL_FEATURES]
    train_context_matrix = cluster_preprocessor.fit_transform(X_train_context)
    test_context_matrix = cluster_preprocessor.transform(X_test_context)
    kmeans = KMeans(n_clusters=N_DEMAND_PROFILES, random_state=RANDOM_STATE, n_init=20)
    train_cluster_ids = kmeans.fit_predict(train_context_matrix)
    test_cluster_ids = kmeans.predict(test_context_matrix)
    cluster_means = pd.DataFrame({'cluster_id': train_cluster_ids, 'bike_demand': train_df['bike_demand'].to_numpy()}).groupby('cluster_id')['bike_demand'].mean().sort_values()
    cluster_stats_df = pd.DataFrame({'cluster_id': train_cluster_ids, 'bike_demand': train_df['bike_demand'].to_numpy()}).groupby('cluster_id')['bike_demand']
    cluster_stds = cluster_stats_df.std()
    sorted_cluster_ids = cluster_means.index.tolist()
    demand_level_labels = ['Low demand profile', 'Moderate demand profile', 'High demand profile', 'Peak demand profile']
    context_labels = ['Contextual demand profile 1', 'Contextual demand profile 2', 'Contextual demand profile 3', 'Contextual demand profile 4']
    all_well_separated = True
    for i in range(len(sorted_cluster_ids) - 1):
        c_a = sorted_cluster_ids[i]
        c_b = sorted_cluster_ids[i + 1]
        mean_diff = abs(cluster_means[c_b] - cluster_means[c_a])
        pooled_std = np.sqrt((cluster_stds[c_a] ** 2 + cluster_stds[c_b] ** 2) / 2)
        d = mean_diff / pooled_std if pooled_std > 0 else 0
        if d < 0.2:
            all_well_separated = False
            break
    if all_well_separated:
        profile_names_ordered = demand_level_labels
        label_style = 'demand-level'
    else:
        profile_names_ordered = context_labels
        label_style = 'contextual'
    cluster_label_map = {int(cluster_id): profile_names_ordered[rank] for rank, cluster_id in enumerate(sorted_cluster_ids)}
    train_df['demand_profile'] = [cluster_label_map[int(x)] for x in train_cluster_ids]
    test_df['demand_profile'] = [cluster_label_map[int(x)] for x in test_cluster_ids]
    cluster_detail_stats = {}
    for cid in sorted_cluster_ids:
        mask = train_cluster_ids == cid
        vals = train_df['bike_demand'].to_numpy()[mask]
        label = cluster_label_map[cid]
        cluster_detail_stats[label] = {'cluster_id': int(cid), 'count': int(len(vals)), 'pct_of_training': round(float(len(vals) / len(train_df) * 100), 1), 'mean': round(float(vals.mean()), 2), 'median': round(float(np.median(vals)), 1), 'std': round(float(vals.std()), 2), 'min': int(vals.min()), 'max': int(vals.max()), 'p10': round(float(np.percentile(vals, 10)), 1), 'p25': round(float(np.percentile(vals, 25)), 1), 'p75': round(float(np.percentile(vals, 75)), 1), 'p90': round(float(np.percentile(vals, 90)), 1)}
    print(f'\n── Discovered Demand Profiles (labels: {label_style}) ──')
    cluster_summary = train_df.groupby('demand_profile')['bike_demand'].agg(['count', 'mean', 'median', 'std']).reindex(profile_names_ordered)
    for profile in profile_names_ordered:
        row = cluster_summary.loc[profile]
        print(f"  {profile:35s}  hours={int(row['count']):>5,}  mean={row['mean']:>7.1f}  median={row['median']:>6.1f}  std={row['std']:>6.1f}")
    MODEL_NUMERIC_FEATURES = CLUSTER_NUMERIC_FEATURES.copy()
    MODEL_CATEGORICAL_FEATURES = ['weather_code', 'season', 'demand_profile']
    MODEL_FEATURES = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
    model_preprocessor = ColumnTransformer(transformers=[('numeric', 'passthrough', MODEL_NUMERIC_FEATURES), ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), MODEL_CATEGORICAL_FEATURES)], remainder='drop', sparse_threshold=0.0)
    model = Pipeline(steps=[('preprocessor', model_preprocessor), ('regressor', HistGradientBoostingRegressor(loss='squared_error', learning_rate=0.06, max_iter=300, max_leaf_nodes=20, min_samples_leaf=25, l2_regularization=2.0, early_stopping=True, random_state=RANDOM_STATE))])
    model.fit(train_df[MODEL_FEATURES], train_df['bike_demand'])
    model_predictions = np.maximum(0, model.predict(test_df[MODEL_FEATURES]))
    baseline_by_dow_hour = train_df.groupby(['day_of_week', 'hour'])['bike_demand'].mean().to_dict()
    overall_training_mean = float(train_df['bike_demand'].mean())
    baseline_predictions = np.array([baseline_by_dow_hour.get((int(row.day_of_week), int(row.hour)), overall_training_mean) for row in test_df.itertuples()], dtype=float)
    actual = test_df['bike_demand'].to_numpy(dtype=float)

    def regression_metrics(y_true, y_pred):
        return {'MAE (trips/hour)': mean_absolute_error(y_true, y_pred), 'RMSE (trips/hour)': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'R2': r2_score(y_true, y_pred)}
    ml_metrics = regression_metrics(actual, model_predictions)
    bl_metrics = regression_metrics(actual, baseline_predictions)
    metrics = pd.DataFrame({'ML model': ml_metrics, 'Weekday-hour baseline': bl_metrics}).T
    print('\n── Evaluation on Held-Out Test Period ──')
    print(f"{'':30s}  {'MAE':>10s}  {'RMSE':>10s}  {'R²':>10s}")
    print('-' * 65)
    for label in ['ML model', 'Weekday-hour baseline']:
        row = metrics.loc[label]
        print(f"  {label:28s}  {row['MAE (trips/hour)']:>10.3f}  {row['RMSE (trips/hour)']:>10.3f}  {row['R2']:>10.3f}")
    print('\nLower MAE/RMSE is better.  R² is descriptive; negative means worse than predicting the test mean.')
    train_bike_rows = bike_rows[bike_rows['hour_start'] < split_timestamp].copy()
    if train_bike_rows.empty:
        raise ValueError('No bicycle trips in the training period.')
    mean_trip_duration_minutes = float(train_bike_rows['duration'].clip(lower=0).mean() / 60.0)
    median_trip_duration_minutes = float(train_bike_rows['duration'].clip(lower=0).median() / 60.0)
    train_bike_rows['day_of_week'] = train_bike_rows['hour_start'].dt.dayofweek
    train_bike_rows['hour'] = train_bike_rows['hour_start'].dt.hour
    station_name_counts = train_bike_rows.groupby(['start_station_id', 'start_station_name']).size().reset_index(name='trips')
    station_name_lookup = station_name_counts.sort_values('trips', ascending=False).drop_duplicates('start_station_id').set_index('start_station_id')['start_station_name'].to_dict()

    def build_station_share_records(group, top_n=TOP_STATIONS_TO_REPORT):
        counts = group.groupby('start_station_id').size().sort_values(ascending=False)
        total = int(counts.sum())
        records = []
        for station_id, count in counts.head(top_n).items():
            sid = int(station_id)
            records.append({'station_id': sid, 'station_name': str(station_name_lookup.get(sid, f'Station {sid}')), 'share': float(count / total) if total else 0.0})
        return records
    station_share_lookup = {}
    for (dow, hour_value), group in train_bike_rows.groupby(['day_of_week', 'hour']):
        station_share_lookup[int(dow), int(hour_value)] = build_station_share_records(group)
    global_station_shares = build_station_share_records(train_bike_rows)
    station_total_count = int(train_bike_rows['start_station_id'].nunique())
    print(f'\nMean trip duration (train): {mean_trip_duration_minutes:.2f} minutes')
    print(f'Stations in training set : {station_total_count}')
    holiday_lookup = {str(d.date()): int(v) for d, v in daily_holiday.items()}
    known_weather_codes = sorted((int(x) for x in hourly['weather_code'].unique()))
    known_season_codes = sorted((int(x) for x in hourly['season'].unique()))
    print(f'Known weather codes      : {known_weather_codes}')
    print(f'Known season codes       : {known_season_codes}')
    print(f'Holiday dates in dataset : {sum((1 for v in holiday_lookup.values() if v == 1))}')
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cluster_demand_means = {int(cluster_id): float(mean_value) for cluster_id, mean_value in cluster_means.items()}
    artifact = {'artifact_version': 2, 'project_name': 'Smart Mobility Demand Forecasting and Capacity Allocation', 'target_name': 'hourly_bicycle_trip_starts', 'bike_vehicle_types': sorted(BIKE_TYPES), 'model': model, 'cluster_preprocessor': cluster_preprocessor, 'kmeans': kmeans, 'cluster_label_map': cluster_label_map, 'cluster_demand_means': cluster_demand_means, 'cluster_detail_stats': cluster_detail_stats, 'cluster_label_style': label_style, 'baseline_by_dow_hour': baseline_by_dow_hour, 'overall_training_mean': overall_training_mean, 'mean_trip_duration_minutes': mean_trip_duration_minutes, 'median_trip_duration_minutes': median_trip_duration_minutes, 'safety_buffer': SAFETY_BUFFER, 'station_share_lookup': station_share_lookup, 'global_station_shares': global_station_shares, 'station_count': station_total_count, 'model_numeric_features': MODEL_NUMERIC_FEATURES, 'model_categorical_features': MODEL_CATEGORICAL_FEATURES, 'cluster_numeric_features': CLUSTER_NUMERIC_FEATURES, 'cluster_categorical_features': CLUSTER_CATEGORICAL_FEATURES, 'training_start': str(train_df.index.min()), 'training_end': str(train_df.index.max()), 'dataset_start': str(hourly.index.min()), 'dataset_end': str(hourly.index.max()), 'test_start': str(test_df.index.min()), 'test_end': str(test_df.index.max()), 'metrics': metrics.to_dict(orient='index'), 'source_dataset_name': DATASET_PATH.name, 'holiday_lookup': holiday_lookup, 'known_weather_codes': known_weather_codes, 'known_season_codes': known_season_codes}
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f'\n── Artifact Saved ──')
    print(f'Path : {ARTIFACT_PATH.resolve()}')
    print(f'Size : {ARTIFACT_PATH.stat().st_size / 1024:.1f} KB')
    print('\nTraining complete. Run the API with:')
    print('  python -m uvicorn main:app --reload')
if __name__ == '__main__':
    main()
