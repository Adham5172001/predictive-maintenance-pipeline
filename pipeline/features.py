"""Predictive Maintenance Feature Engineering — Author: Adham Aboulkheir"""
import numpy as np
import pandas as pd
from scipy import stats

def extract_features(signal):
    return {
        "mean": float(np.mean(signal)), "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "peak": float(np.max(np.abs(signal))),
        "kurtosis": float(stats.kurtosis(signal)),
        "skewness": float(stats.skew(signal)),
        "spectral_energy": float(np.sum(np.abs(np.fft.rfft(signal))**2)),
    }

def create_rolling_features(df, sensor_cols, windows=[10, 30, 60]):
    result = df.copy()
    for col in sensor_cols:
        if col not in df.columns: continue
        for w in windows:
            result[f"{col}_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            result[f"{col}_std_{w}"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
    return result

def generate_sensor_data(n_samples=5000, failure_at=4500, seed=42):
    np.random.seed(seed)
    t = np.arange(n_samples)
    deg = np.where(t > failure_at - 500, np.linspace(0, 3, n_samples)[np.maximum(t-(failure_at-500), 0)], 0)
    return pd.DataFrame({
        "timestamp": t,
        "vibration_x": 1.0 + 0.002*t + deg + np.random.normal(0, 0.1, n_samples),
        "vibration_y": 0.8 + 0.001*t + deg*0.8 + np.random.normal(0, 0.08, n_samples),
        "temperature": 65 + 0.005*t + deg*5 + np.random.normal(0, 1.5, n_samples),
        "pressure": 4.0 + 0.001*t + deg*0.3 + np.random.normal(0, 0.1, n_samples),
        "rpm": 1450 + np.random.normal(0, 20, n_samples) - deg*50,
        "failure": (t >= failure_at).astype(int)
    })

if __name__ == "__main__":
    df = generate_sensor_data()
    sensor_cols = ["vibration_x", "vibration_y", "temperature", "pressure", "rpm"]
    df_feat = create_rolling_features(df, sensor_cols)
    print(f"Features shape: {df_feat.shape}")
    print(f"Failure rate: {df['failure'].mean():.1%}")
