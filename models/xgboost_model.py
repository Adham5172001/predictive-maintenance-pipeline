"""
XGBoost Model for Predictive Maintenance
Author: Adham Aboulkheir
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    random_state: int = 42


class XGBoostMaintenanceModel:
    """
    XGBoost-based failure prediction model.
    Note: Uses sklearn GradientBoosting as XGBoost proxy.
    Install xgboost for production: pip install xgboost
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            random_state=self.config.random_state
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostMaintenanceModel":
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        return {
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "auc_roc": roc_auc_score(y, y_prob),
            "feature_importances": self.model.feature_importances_
        }

    def get_failure_probability(self, X: np.ndarray,
                                 lead_time_hours: int = 48) -> dict:
        """Get failure probability with lead time estimate."""
        proba = self.predict_proba(X)[:, 1]
        return {
            "failure_probability": float(proba.mean()),
            "high_risk_samples": int((proba > 0.7).sum()),
            "lead_time_hours": lead_time_hours,
            "recommendation": "Immediate maintenance" if proba.mean() > 0.7
                               else "Schedule maintenance" if proba.mean() > 0.4
                               else "Normal operation"
        }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.features import generate_sensor_data, create_rolling_features
    from sklearn.model_selection import train_test_split

    print("XGBoost Predictive Maintenance Demo")
    df = generate_sensor_data(n_samples=5000, failure_at=4500)
    sensor_cols = ["vibration_x", "vibration_y", "temperature", "pressure", "rpm"]
    df_feat = create_rolling_features(df, sensor_cols)
    feature_cols = [c for c in df_feat.columns if c not in ["timestamp", "failure"]]
    X = df_feat[feature_cols].fillna(0).values
    y = df["failure"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBoostMaintenanceModel()
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    print(f"F1: {metrics['f1_weighted']:.4f} | AUC-ROC: {metrics['auc_roc']:.4f}")
    risk = model.get_failure_probability(X_test)
    print(f"Mean failure probability: {risk['failure_probability']:.1%}")
    print(f"Recommendation: {risk['recommendation']}")
