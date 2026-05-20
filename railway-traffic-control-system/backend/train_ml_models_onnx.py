"""
AI-Powered Railway Traffic Control System
ML Model Training Pipeline - ONNX Version
"""

import logging
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

# ONNX imports
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=10000):
    """Generate a rich synthetic dataset for railway traffic."""
    logger.info(f"Generating {n_samples} synthetic records...")
    np.random.seed(42)

    data = {
        'trains_in_section': np.random.randint(5, 50, n_samples),
        'available_platforms': np.random.randint(2, 6, n_samples),
        'platform_utilization': np.random.uniform(20, 95, n_samples),
        'weather_severity': np.random.uniform(0, 1, n_samples),
        'rainfall_mm': np.random.uniform(0, 50, n_samples),
        'fog_intensity': np.random.uniform(0, 1, n_samples),
        'temperature_c': np.random.uniform(-10, 35, n_samples),
        'is_peak_hour': np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)

    # Logic for conflict probability
    # Higher trains, high utilization, bad weather -> higher conflict risk
    conflict_score = (
        df['trains_in_section'] * 0.02 +
        df['platform_utilization'] * 0.01 +
        df['weather_severity'] * 0.2 +
        df['is_peak_hour'] * 0.15
    )
    df['conflict_occurred'] = (conflict_score + np.random.normal(0, 0.1, n_samples) > 1.0).astype(int)

    # Logic for delay minutes
    # Delay depends on conflicts, weather, and congestion
    df['delay_minutes'] = (
        df['conflict_occurred'] * 15 +
        df['trains_in_section'] * 0.5 +
        df['weather_severity'] * 10 +
        np.random.normal(5, 2, n_samples)
    ).clip(lower=0)

    return df

def train_and_export_onnx():
    output_dir = 'railway-traffic-control-system/backend/models'
    os.makedirs(output_dir, exist_ok=True)

    df = generate_synthetic_data()

    features = [
        'trains_in_section', 'available_platforms', 'platform_utilization',
        'weather_severity', 'rainfall_mm', 'fog_intensity', 'temperature_c',
        'is_peak_hour'
    ]

    X = df[features].astype(np.float32)
    y_conflict = df['conflict_occurred']
    y_delay = df['delay_minutes'].astype(np.float32)

    # --- Conflict Model ---
    logger.info("Training Conflict Detector...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_conflict, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    logger.info(f"Conflict Detector Accuracy: {acc:.4f}")

    # Export to ONNX
    initial_type = [('float_input', FloatTensorType([None, len(features)]))]
    onx_clf = convert_sklearn(clf, initial_types=initial_type, target_opset=12)
    with open(os.path.join(output_dir, "conflict_detector.onnx"), "wb") as f:
        f.write(onx_clf.SerializeToString())
    logger.info("Saved conflict_detector.onnx")

    # --- Delay Model ---
    logger.info("Training Delay Predictor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_delay, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    reg.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, reg.predict(X_test))
    logger.info(f"Delay Predictor MAE: {mae:.4f}")

    # Export to ONNX
    onx_reg = convert_sklearn(reg, initial_types=initial_type, target_opset=12)
    with open(os.path.join(output_dir, "delay_predictor.onnx"), "wb") as f:
        f.write(onx_reg.SerializeToString())
    logger.info("Saved delay_predictor.onnx")

    # --- Metadata ---
    metadata = {
        'training_date': datetime.now(timezone.utc).isoformat(),
        'features': features,
        'metrics': {
            'conflict_accuracy': acc,
            'delay_mae': mae
        }
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save as pkl for legacy support if needed (optional, but good practice)
    joblib.dump(clf, os.path.join(output_dir, "conflict_detector.pkl"))
    joblib.dump(reg, os.path.join(output_dir, "delay_predictor.pkl"))

if __name__ == '__main__':
    train_and_export_onnx()
