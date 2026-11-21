from __future__ import annotations
import math
import os
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, PyMongoError
from bson import ObjectId
from bson.errors import InvalidId
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DATABASE = os.environ.get('MONGODB_DATABASE', 'smart_mobility')
MONGODB_COLLECTION = os.environ.get('MONGODB_COLLECTION', 'prediction_history')
ARTIFACT_PATH = Path('artifacts') / 'bike_demand_model.pkl'
SAFETY_BUFFER = 1.2
BASELINE_THRESHOLD = 0.15
app = FastAPI(title='Smart Mobility Demand Forecasting API', description='Predicts hourly shared-bicycle demand using weather, time, and seasonal context.  Returns demand profiles, capacity recommendations, and station pickup allocation estimates.  Prediction history is persisted to MongoDB.', version='1.2.2')
artifact: Optional[Dict[str, Any]] = None
mongo_client: Optional[MongoClient] = None
mongo_collection = None

def format_utc_timestamp(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if s.endswith('Z'):
            return s
        try:
            parsed = datetime.fromisoformat(s)
            val = parsed
        except Exception:
            return s
    if isinstance(val, datetime):
        if val.tzinfo is not None:
            val = val.astimezone(timezone.utc)
        else:
            val = val.replace(tzinfo=timezone.utc)
        ms = val.microsecond // 1000
        return val.strftime(f'%Y-%m-%dT%H:%M:%S.{ms:03d}Z')
    return str(val)

@app.on_event('startup')
def startup() -> None:
    global artifact, mongo_client, mongo_collection
    if ARTIFACT_PATH.exists():
        artifact = joblib.load(ARTIFACT_PATH)
        print(f'Model artifact loaded from {ARTIFACT_PATH.resolve()}')
    else:
        print(f'WARNING: Model artifact not found at {ARTIFACT_PATH.resolve()}.  Run `python train_model.py` first.')
    try:
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000, tz_aware=True)
        mongo_client.admin.command('ping')
        db = mongo_client[MONGODB_DATABASE]
        mongo_collection = db[MONGODB_COLLECTION]
        mongo_collection.create_index([('created_at', DESCENDING)])
        print(f'MongoDB connected — database: {MONGODB_DATABASE}, collection: {MONGODB_COLLECTION}')
    except ConnectionFailure as exc:
        print(f'WARNING: MongoDB connection failed — {exc}')
        mongo_client = None
        mongo_collection = None

@app.on_event('shutdown')
def shutdown() -> None:
    global mongo_client, mongo_collection
    if mongo_client is not None:
        mongo_client.close()
        mongo_client = None
        mongo_collection = None
        print('MongoDB connection closed.')

def _mongo_is_available() -> bool:
    if mongo_client is None:
        return False
    try:
        mongo_client.admin.command('ping')
        return True
    except PyMongoError:
        return False

class PredictRequest(BaseModel):
    date: str = Field(..., description='Date in YYYY-MM-DD format', examples=['2023-11-07'])
    hour: int = Field(..., ge=0, le=23, description='Hour of day (0-23)')
    temperature: float = Field(..., description='Temperature (°C)')
    max_temperature: float = Field(..., description='Daily maximum temperature (°C)')
    min_temperature: float = Field(..., description='Daily minimum temperature (°C)')
    wind_speed: float = Field(..., ge=0, description='Wind speed (m/s)')
    precipitation: float = Field(..., ge=0, description='Precipitation (mm)')
    humidity: float = Field(..., ge=0, le=100, description='Relative humidity (0-100%)')
    weather: int = Field(..., description='Weather code (integer)')
    sunshine: float = Field(..., ge=0, description='Sunshine duration (minutes)')
    season: int = Field(..., description='Season code (integer)')
    is_holiday: bool = Field(..., description='Whether the date is a public holiday')

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('date must be in YYYY-MM-DD format')
        return v

    @field_validator('temperature', 'max_temperature', 'min_temperature', 'wind_speed', 'precipitation', 'humidity', 'sunshine', mode='after')
    @classmethod
    def validate_finite_numbers(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError('Numerical values must be finite and cannot be NaN or infinite')
        return v

    @model_validator(mode='after')
    def validate_temperature_consistency(self) -> 'PredictRequest':
        if self.min_temperature > self.max_temperature:
            raise ValueError(f'min_temperature ({self.min_temperature}) must not exceed max_temperature ({self.max_temperature})')
        if self.temperature < self.min_temperature:
            raise ValueError(f'temperature ({self.temperature}) must not be below min_temperature ({self.min_temperature})')
        if self.temperature > self.max_temperature:
            raise ValueError(f'temperature ({self.temperature}) must not exceed max_temperature ({self.max_temperature})')
        return self

def build_features(req: PredictRequest) -> tuple[pd.Timestamp, int, int, dict]:
    timestamp = pd.Timestamp(req.date).normalize() + pd.Timedelta(hours=req.hour)
    dow = int(timestamp.dayofweek)
    month_val = int(timestamp.month)
    doy = int(timestamp.dayofyear)
    row = {'temperature': float(req.temperature), 'max_temperature': float(req.max_temperature), 'min_temperature': float(req.min_temperature), 'wind_speed': float(req.wind_speed), 'precipitation': float(req.precipitation), 'humidity': float(req.humidity), 'sunshine': float(req.sunshine), 'hour_sin': math.sin(2 * math.pi * req.hour / 24), 'hour_cos': math.cos(2 * math.pi * req.hour / 24), 'dow_sin': math.sin(2 * math.pi * dow / 7), 'dow_cos': math.cos(2 * math.pi * dow / 7), 'month_sin': math.sin(2 * math.pi * (month_val - 1) / 12), 'month_cos': math.cos(2 * math.pi * (month_val - 1) / 12), 'day_of_year_sin': math.sin(2 * math.pi * (doy - 1) / 365.25), 'day_of_year_cos': math.cos(2 * math.pi * (doy - 1) / 365.25), 'is_holiday': int(bool(req.is_holiday)), 'is_weekend': int(dow >= 5), 'weather_code': str(int(req.weather)), 'season': str(int(req.season))}
    return (timestamp, dow, req.hour, row)

def classify_warnings(timestamp: pd.Timestamp, req: PredictRequest, art: Dict[str, Any]) -> List[str]:
    warnings_list: List[str] = []
    training_start = art.get('training_start', '')
    training_end = art.get('training_end', '')
    dataset_start = art.get('dataset_start', '')
    dataset_end = art.get('dataset_end', '')
    try:
        ts_train_start = pd.Timestamp(training_start)
        ts_train_end = pd.Timestamp(training_end)
        ts_ds_start = pd.Timestamp(dataset_start)
        ts_ds_end = pd.Timestamp(dataset_end)
        if timestamp < ts_ds_start or timestamp > ts_ds_end:
            warnings_list.append(f'The requested date/hour ({timestamp}) is outside the original dataset coverage ({dataset_start} to {dataset_end}). The prediction is an extrapolation and may be less reliable.')
        elif timestamp < ts_train_start or timestamp > ts_train_end:
            warnings_list.append(f'The requested date/hour ({timestamp}) belongs to the held-out evaluation period. This observation was not used for model training. It is within the original dataset coverage.')
    except Exception:
        pass
    holiday_lookup = art.get('holiday_lookup', {})
    date_str = req.date
    if date_str in holiday_lookup:
        known_holiday = bool(holiday_lookup[date_str])
        if req.is_holiday != known_holiday:
            expected = 'a holiday' if known_holiday else 'not a holiday'
            supplied = 'true' if req.is_holiday else 'false'
            raise ValueError(f'Holiday mismatch for {date_str}: the dataset records this date as {expected}, but is_holiday was supplied as {supplied}.')
    known_weather = art.get('known_weather_codes', [])
    if known_weather and req.weather not in known_weather:
        warnings_list.append(f'Weather code {req.weather} was not observed in the training dataset. Known codes: {known_weather}. The model may handle this as an unseen category.')
    known_seasons = art.get('known_season_codes', [])
    if known_seasons and req.season not in known_seasons:
        warnings_list.append(f'Season code {req.season} was not observed in the training dataset. Known codes: {known_seasons}. The model may handle this as an unseen category.')
    return warnings_list

@app.get('/', tags=['Info'], summary='Project information and API availability')
def root():
    return {'project': 'Smart Mobility Demand Forecasting and Capacity Allocation', 'description': 'Predicts hourly shared-bicycle demand and provides capacity recommendations using weather, time, and seasonal context.', 'api_version': '1.2.2', 'model_loaded': artifact is not None, 'mongodb_connected': _mongo_is_available(), 'endpoints': {'GET /': 'Project information (this response)', 'GET /health': 'API and model health check', 'GET /docs': 'Interactive API documentation (Swagger UI)', 'POST /predict': 'Bicycle demand prediction with capacity recommendation', 'GET /history': 'Paginated prediction history', 'GET /history/{record_id}': 'Retrieve a single prediction record'}}

@app.get('/health', tags=['Info'], summary='API health check including model and MongoDB status')
def health():
    model_ok = artifact is not None
    mongo_ok = _mongo_is_available()
    overall_healthy = model_ok and mongo_ok
    response: Dict[str, Any] = {'status': 'healthy' if overall_healthy else 'degraded', 'model_loaded': model_ok, 'mongodb': {'connected': mongo_ok, 'database': MONGODB_DATABASE, 'collection': MONGODB_COLLECTION}}
    if not model_ok:
        response['model_message'] = 'Model artifact not loaded. Run `python train_model.py` first.'
    if not mongo_ok:
        response['mongodb_message'] = 'MongoDB is not reachable. Predictions will not be persisted.'
    if model_ok:
        response.update({'artifact_version': artifact.get('artifact_version'), 'training_data_range': {'start': artifact.get('training_start'), 'end': artifact.get('training_end')}, 'dataset_range': {'start': artifact.get('dataset_start'), 'end': artifact.get('dataset_end')}, 'test_period': {'start': artifact.get('test_start'), 'end': artifact.get('test_end')}, 'target': artifact.get('target_name'), 'bike_vehicle_types': artifact.get('bike_vehicle_types'), 'stations_in_training': artifact.get('station_count')})
    return response

@app.post('/predict', tags=['Prediction'], summary='Predict hourly bicycle demand and capacity recommendations', description='Accepts 12 input fields, derives cyclic features, assigns contextual profile, predicts demand, and stores history in MongoDB.')
def predict(req: PredictRequest):
    if artifact is None:
        raise HTTPException(status_code=503, detail='Model not loaded. Run `python train_model.py` and restart the server.')
    timestamp, dow, hour_val, feature_row = build_features(req)
    try:
        warnings_list = classify_warnings(timestamp, req, artifact)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    cluster_features = artifact['cluster_numeric_features'] + artifact['cluster_categorical_features']
    context_df = pd.DataFrame([feature_row])
    context_matrix = artifact['cluster_preprocessor'].transform(context_df[cluster_features])
    cluster_id = int(artifact['kmeans'].predict(context_matrix)[0])
    demand_profile = artifact['cluster_label_map'][cluster_id]
    feature_row['demand_profile'] = demand_profile
    model_features = artifact['model_numeric_features'] + artifact['model_categorical_features']
    feature_df = pd.DataFrame([feature_row])
    raw_prediction = float(artifact['model'].predict(feature_df[model_features])[0])
    predicted_demand = round(max(0.0, raw_prediction), 2)
    baseline_demand = round(artifact['baseline_by_dow_hour'].get((dow, hour_val), artifact['overall_training_mean']), 2)
    mean_duration = artifact['mean_trip_duration_minutes']
    safety_buffer = artifact.get('safety_buffer', SAFETY_BUFFER)
    estimated_concurrent = round(predicted_demand * mean_duration / 60.0, 2)
    recommended_capacity = round(estimated_concurrent * safety_buffer, 2)
    if baseline_demand > 0:
        pct_diff = (predicted_demand - baseline_demand) / baseline_demand
    else:
        pct_diff = 0.0 if predicted_demand == 0 else 1.0
    if pct_diff >= BASELINE_THRESHOLD:
        recommendation = 'Increase bicycle availability'
    elif pct_diff <= -BASELINE_THRESHOLD:
        recommendation = 'Reduce excess bicycle availability'
    else:
        recommendation = 'Maintain current bicycle availability'
    profile_avg_demand = round(artifact['cluster_demand_means'].get(cluster_id, 0.0), 2)
    profile_stats = None
    cluster_detail = artifact.get('cluster_detail_stats', {})
    if demand_profile in cluster_detail:
        profile_stats = cluster_detail[demand_profile]
    station_shares = artifact['station_share_lookup'].get((dow, hour_val), artifact['global_station_shares'])
    station_allocation = []
    allocated_share = 0.0
    for s in station_shares:
        expected_pickups = round(predicted_demand * s['share'], 2)
        station_allocation.append({'station_id': s['station_id'], 'station_name': s['station_name'], 'historical_share_pct': round(s['share'] * 100, 2), 'expected_pickups': expected_pickups})
        allocated_share += s['share']
    remaining_share = max(0.0, 1.0 - allocated_share)
    remaining_pickups = round(predicted_demand * remaining_share, 2)
    prediction_response: Dict[str, Any] = {'input': {'date': req.date, 'hour': req.hour, 'day_of_week': dow, 'day_name': timestamp.day_name(), 'is_weekend': dow >= 5}, 'prediction': {'predicted_hourly_bicycle_trips': predicted_demand, 'demand_profile': demand_profile, 'profile_average_demand': profile_avg_demand, 'profile_description': "The demand profile represents contextual similarity (time, weather, season) discovered by K-Means clustering. The predicted trips are from the regression model and may differ from the profile's historical average."}, 'baseline': {'weekday_hour_baseline_demand': baseline_demand, 'difference_from_baseline_pct': round(pct_diff * 100, 2)}, 'capacity': {'recommendation': recommendation, 'estimated_concurrent_bicycles': estimated_concurrent, 'recommended_concurrent_capacity': recommended_capacity, 'mean_trip_duration_minutes': round(mean_duration, 2), 'safety_buffer_pct': round((safety_buffer - 1) * 100)}, 'station_allocation': {'method': 'Historical share-based allocation. Station pickups are estimated by distributing the city-wide predicted demand using weekday/hour pickup shares from the training period. This is not a separate station-level ML prediction.', 'top_stations': station_allocation, 'remaining_stations': {'share_pct': round(remaining_share * 100, 2), 'expected_pickups': remaining_pickups}}, 'training_period': {'start': artifact.get('training_start', ''), 'end': artifact.get('training_end', '')}, 'warnings': warnings_list}
    if profile_stats is not None:
        prediction_response['prediction']['profile_statistics'] = profile_stats
    now = datetime.now(timezone.utc)
    created_at = now.replace(microsecond=now.microsecond // 1000 * 1000)
    created_at_iso = format_utc_timestamp(created_at)
    record_oid = ObjectId()
    record_id = str(record_oid)
    prediction_response['prediction_record_id'] = record_id
    prediction_response['created_at'] = created_at_iso
    document = {'_id': record_oid, 'request_input': req.model_dump(), 'response': prediction_response, 'created_at': created_at}
    if mongo_collection is None or not _mongo_is_available():
        raise HTTPException(status_code=503, detail='MongoDB is unavailable. The prediction was computed but could not be saved. Please ensure MongoDB is running and restart the server.')
    try:
        mongo_collection.insert_one(document)
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f'Failed to save prediction to MongoDB: {exc}')
    return prediction_response

@app.get('/history', tags=['History'], summary='Retrieve paginated prediction history (latest first)')
def get_history(limit: int=Query(default=20, ge=1, le=100, description='Records per page'), offset: int=Query(default=0, ge=0, description='Number of records to skip')):
    if mongo_collection is None or not _mongo_is_available():
        raise HTTPException(status_code=503, detail='MongoDB is unavailable. Cannot retrieve prediction history.')
    try:
        total = mongo_collection.count_documents({})
        cursor = mongo_collection.find().sort('created_at', DESCENDING).skip(offset).limit(limit)
        records = []
        for doc in cursor:
            resp = doc.get('response', {})
            if resp and resp.get('warnings') is None:
                resp['warnings'] = []
            created_iso = format_utc_timestamp(doc.get('created_at'))
            if resp:
                if not resp.get('created_at'):
                    resp['created_at'] = created_iso
                else:
                    resp['created_at'] = format_utc_timestamp(resp['created_at'])
                if not resp.get('prediction_record_id'):
                    resp['prediction_record_id'] = str(doc['_id'])
            records.append({'record_id': str(doc['_id']), 'created_at': created_iso, 'input': doc.get('request_input'), 'prediction_response': resp})
        return {'total': total, 'limit': limit, 'offset': offset, 'records': records}
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f'Failed to retrieve prediction history: {exc}')

@app.get('/history/{record_id}', tags=['History'], summary='Retrieve a single prediction record by MongoDB ObjectId')
def get_history_record(record_id: str):
    if mongo_collection is None or not _mongo_is_available():
        raise HTTPException(status_code=503, detail='MongoDB is unavailable. Cannot retrieve prediction record.')
    try:
        oid = ObjectId(record_id)
    except (InvalidId, Exception):
        raise HTTPException(status_code=400, detail=f"Invalid record ID format: '{record_id}'. Must be a 24-character hex string.")
    try:
        doc = mongo_collection.find_one({'_id': oid})
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f'Database error while retrieving record: {exc}')
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Prediction record not found: '{record_id}'.")
    resp = doc.get('response', {})
    if resp and resp.get('warnings') is None:
        resp['warnings'] = []
    created_iso = format_utc_timestamp(doc.get('created_at'))
    if resp:
        if not resp.get('created_at'):
            resp['created_at'] = created_iso
        else:
            resp['created_at'] = format_utc_timestamp(resp['created_at'])
        if not resp.get('prediction_record_id'):
            resp['prediction_record_id'] = str(doc['_id'])
    return {'record_id': str(doc['_id']), 'created_at': created_iso, 'input': doc.get('request_input'), 'prediction_response': resp}
