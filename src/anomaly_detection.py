import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/processed/community_clusters.csv"
OUTPUT_PATH = "outputs/anomaly_results.csv"


df = pd.read_csv(INPUT_PATH)

features = df[[
    "total_crime_count",
    "total_crime_weight",
    "safety_score"
]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

model = IsolationForest(
    contamination=0.10,
    random_state=42
)

df["anomaly_label"] = model.fit_predict(scaled_features)

df["anomaly_status"] = df["anomaly_label"].map({
    1: "Normal",
    -1: "Anomaly"
})

Path("outputs").mkdir(exist_ok=True)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("Anomaly Detection Sonuçları")
print("---------------------------")
print(df[[
    "community_area",
    "safety_score",
    "risk_level",
    "total_crime_count",
    "total_crime_weight",
    "anomaly_status"
]].head(20))

print(f"\nAnomaly sonuçları kaydedildi: {OUTPUT_PATH}")