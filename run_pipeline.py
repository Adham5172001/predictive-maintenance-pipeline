"""Predictive Maintenance Pipeline Demo — Author: Adham Aboulkheir"""
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from pipeline.features import generate_sensor_data, create_rolling_features
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

def main():
    print("Predictive Maintenance Pipeline Demo")
    os.makedirs("outputs", exist_ok=True)
    df = generate_sensor_data(n_samples=5000, failure_at=4500)
    sensor_cols = ["vibration_x", "vibration_y", "temperature", "pressure", "rpm"]
    df_feat = create_rolling_features(df, sensor_cols)
    feature_cols = [c for c in df_feat.columns if c not in ["timestamp", "failure"]]
    X = df_feat[feature_cols].fillna(0).values
    y = df["failure"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(scaler.fit_transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))
    y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
    print(f"  F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")
    axes[0].plot(df["timestamp"], df["vibration_x"], color="#00c9b1", linewidth=0.5, alpha=0.7)
    axes[0].axvline(x=4500, color="#ff7b72", linestyle="--", linewidth=2, label="Failure")
    axes[0].set_title("Vibration Signal with Failure Point", color="white")
    axes[0].set_xlabel("Time", color="white")
    axes[0].legend(facecolor="#161b22", labelcolor="white", fontsize=8)
    axes[0].tick_params(colors="white")
    axes[0].grid(alpha=0.3, color="#21262d")
    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])[:8]
    names = [f[0] for f in importances]
    imps = [f[1] for f in importances]
    axes[1].barh(names[::-1], imps[::-1], color="#00c9b1", alpha=0.85)
    axes[1].set_title("Feature Importance", color="white")
    axes[1].set_xlabel("Importance", color="white")
    axes[1].tick_params(colors="white")
    axes[1].grid(axis="x", alpha=0.3, color="#21262d")
    plt.tight_layout()
    plt.savefig("outputs/predictive_maintenance_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved: outputs/predictive_maintenance_results.png")

if __name__ == "__main__":
    main()
