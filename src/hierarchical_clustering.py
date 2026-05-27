import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/processed/community_clusters.csv"
OUTPUT_PATH = "data/processed/hierarchical_clusters.csv"
METRICS_PATH = "outputs/hierarchical_clustering_metrics.csv"
DENDROGRAM_PATH = "outputs/hierarchical_dendrogram.png"


df = pd.read_csv(INPUT_PATH)

features = df[[
    "total_crime_count",
    "total_crime_weight",
    "safety_score"
]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

df["hierarchical_cluster"] = model.fit_predict(scaled_features)

silhouette = silhouette_score(
    scaled_features,
    df["hierarchical_cluster"]
)

print(f"Hierarchical Clustering Silhouette Score: {silhouette:.4f}")

cluster_means = df.groupby("hierarchical_cluster")["safety_score"].mean()
sorted_clusters = cluster_means.sort_values()

risk_mapping = {
    sorted_clusters.index[0]: "High Risk",
    sorted_clusters.index[1]: "Medium Risk",
    sorted_clusters.index[2]: "Safe"
}

df["hierarchical_risk_level"] = df["hierarchical_cluster"].map(risk_mapping)

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

metrics_df = pd.DataFrame({
    "Method": ["Hierarchical Clustering"],
    "Metric": ["Silhouette Score"],
    "Value": [round(silhouette, 4)]
})

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
    encoding="utf-8-sig"
)

linked = linkage(
    scaled_features,
    method="ward"
)

plt.figure(figsize=(12, 6))

dendrogram(
    linked,
    labels=df["community_area"].astype(str).values,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Community Area")
plt.ylabel("Distance")

plt.tight_layout()
plt.savefig(DENDROGRAM_PATH)
plt.show()

print(f"Hierarchical clustering sonucu kaydedildi: {OUTPUT_PATH}")
print(f"Hierarchical metriği kaydedildi: {METRICS_PATH}")
print(f"Dendrogram görseli kaydedildi: {DENDROGRAM_PATH}")