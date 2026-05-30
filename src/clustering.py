import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = "data/processed/community_safety_scores.csv"
OUTPUT_PATH = "data/processed/community_clusters.csv"
METRICS_PATH = "outputs/clustering_metrics.csv"

def load_data():
    print(f"Panel veri yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def create_community_profiles(df):
    # Panel veriyi mahalle bazında 5 yıllık ortalamalarla 77 satıra indirgiyoruz
    profile_df = df.groupby("community_area").agg(
        total_crime_count=("total_crime_count", "mean"),
        total_crime_weight=("total_crime_weight", "mean"),
        safety_score=("safety_score", "mean"),
        avg_latitude=("avg_latitude", "first"), # Harita için gerekli koordinatları tutuyoruz
        avg_longitude=("avg_longitude", "first")
    ).reset_index()
    
    print(f"Mahalle profilleri oluşturuldu. Boyut: {profile_df.shape}")
    return profile_df

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
    print(f"\nSilhouette Score (77 Mahalle üzerinden): {silhouette:.4f}")

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
        risk_label = cluster_data["risk_level"].iloc[0] 

        plt.scatter(
            cluster_data["total_crime_weight"],
            cluster_data["safety_score"],
            label=f"{risk_label} (Cluster {cluster})"
        )

    plt.xlabel("Average Total Crime Weight")
    plt.ylabel("Average Safety Score")
    plt.title("Chicago Crime Risk Clusters (5-Year Average Profiles)")
    plt.legend()
    plt.grid(True, alpha=0.3) # Grafiğin okunabilirliğini artırmak için eklendi
    plt.show()

if __name__ == "__main__":
    panel_df = load_data()

    # K-Means için mahallelerin 5 yıllık genel ortalamalarını çıkarıyoruz
    profile_df = create_community_profiles(panel_df)

    # K-Means'i bu 77 genel profil üzerinde uyguluyoruz
    clustered_df, model = apply_kmeans(profile_df)

    clustered_df = label_clusters(clustered_df)

    print("\nİlk 10 Mahalle Risk Profili:")
    print(clustered_df[[
        "community_area",
        "safety_score",
        "risk_level"
    ]].head(10))

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    clustered_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nKümeleme sonucu kaydedildi: {OUTPUT_PATH}")

    visualize_clusters(clustered_df)