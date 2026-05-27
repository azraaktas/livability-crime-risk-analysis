import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_PATH = "data/processed/community_safety_scores.csv"
OUTPUT_PATH = "data/processed/community_clusters.csv"
METRICS_PATH = "outputs/clustering_metrics.csv"


def load_data():
    return pd.read_csv(INPUT_PATH)


def apply_kmeans(df):

    features = df[[
        "total_crime_count",
        "total_crime_weight",
        "safety_score"
    ]]

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["cluster"] = kmeans.fit_predict(features)

    silhouette = silhouette_score(features, df["cluster"])

    print(f"Silhouette Score: {silhouette:.4f}")

    Path("outputs").mkdir(exist_ok=True)

    metrics_df = pd.DataFrame({
        "Metric": ["Silhouette Score"],
        "Value": [round(silhouette, 4)]
    })

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Kümeleme metriği kaydedildi: {METRICS_PATH}")

    return df, kmeans


def label_clusters(df):

    cluster_means = df.groupby("cluster")["safety_score"].mean()

    sorted_clusters = cluster_means.sort_values()

    risk_mapping = {
        sorted_clusters.index[0]: "High Risk",
        sorted_clusters.index[1]: "Medium Risk",
        sorted_clusters.index[2]: "Safe"
    }

    df["risk_level"] = df["cluster"].map(risk_mapping)

    return df


def visualize_clusters(df):

    plt.figure(figsize=(10, 6))

    for cluster in df["cluster"].unique():

        cluster_data = df[df["cluster"] == cluster]

        plt.scatter(
            cluster_data["total_crime_weight"],
            cluster_data["safety_score"],
            label=f"Cluster {cluster}"
        )

    plt.xlabel("Total Crime Weight")
    plt.ylabel("Safety Score")
    plt.title("Crime Risk Clusters")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    df = load_data()

    df, model = apply_kmeans(df)

    df = label_clusters(df)

    print(df[[
        "community_area",
        "safety_score",
        "risk_level"
    ]].head(10))

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Kümeleme sonucu kaydedildi: {OUTPUT_PATH}")

    visualize_clusters(df)