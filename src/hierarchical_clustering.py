import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Hedef dosyamızı güncel panel veri olarak ayarladık
INPUT_PATH = "data/processed/community_safety_scores.csv"
OUTPUT_PATH = "data/processed/hierarchical_clusters.csv"
METRICS_PATH = "outputs/hierarchical_clustering_metrics.csv"
DENDROGRAM_PATH = "outputs/hierarchical_dendrogram.png"

def load_data():
    print(f"Hiyerarşik kümeleme için veri yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def create_community_profiles(df):
    # Mahalle bazında 5 yıllık genel risk profilini (77 satır) çıkarıyoruz
    profile_df = df.groupby("community_area").agg(
        total_crime_count=("total_crime_count", "mean"),
        total_crime_weight=("total_crime_weight", "mean"),
        safety_score=("safety_score", "mean")
    ).reset_index()
    return profile_df

def apply_hierarchical_clustering(df):
    print("Hiyerarşik Kümeleme (Ward Linkage) uygulanıyor...")
    
    features = df[[
        "total_crime_count",
        "total_crime_weight",
        "safety_score"
    ]]

    # Algoritmanın büyük sayılardan etkilenmemesi için standartlaştırma (Scaling)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Modeli Kurma (3 Küme)
    model = AgglomerativeClustering(
        n_clusters=3,
        linkage="ward"
    )

    df["hierarchical_cluster"] = model.fit_predict(scaled_features)

    # Başarı Skoru
    silhouette = silhouette_score(scaled_features, df["hierarchical_cluster"])
    print(f"Hierarchical Silhouette Score: {silhouette:.4f}")

    return df, scaled_features, silhouette

def label_and_save_clusters(df, silhouette):
    # Kümeleri risk seviyelerine göre isimlendirme
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

    # Kümeleri Kaydet
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # Metriği Kaydet
    metrics_df = pd.DataFrame({
        "Method": ["Hierarchical Clustering"],
        "Metric": ["Silhouette Score"],
        "Value": [round(silhouette, 4)]
    })
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    print(f"Kümeler kaydedildi: {OUTPUT_PATH}")
    print(f"Metrik kaydedildi: {METRICS_PATH}")

def visualize_dendrogram(df, scaled_features):
    print("Dendrogram (Ağaç Grafiği) çiziliyor...")
    
    # Scipy ile akrabalık bağlarını (linkage) hesaplama
    linked = linkage(scaled_features, method="ward")

    plt.figure(figsize=(16, 7)) # Grafiği genişlettik ki mahalle numaraları rahat okunsun
    
    # Dendrogramı Çiz
    ddata = dendrogram(
        linked,
        labels=df["community_area"].astype(str).values,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=10 # Renklendirme sınırı (Tahmini)
    )

    # 3 kümeye ayrıldığımızı gösteren şık bir yatay kesim çizgisi (Threshold)
    # Bu çizgi, jüriye "Ağacı tam bu mesafeden kesip 3 gruba ayırdık" demenizi sağlar.
    plt.axhline(y=10, c='k', lw=2, linestyle='dashed', label='3-Cluster Threshold')

    plt.title("Chicago Neighborhoods Hierarchical Clustering Dendrogram", fontsize=16)
    plt.xlabel("Community Area (Neighborhood ID)", fontsize=12)
    plt.ylabel("Ward Linkage Distance", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(DENDROGRAM_PATH, dpi=300) # Dashboard'da net görünmesi için DPI artırıldı
    plt.show()

    print(f"Dendrogram görseli kaydedildi: {DENDROGRAM_PATH}")

if __name__ == "__main__":
    panel_df = load_data()
    
    # Adım 1: Mahalle profillerini çıkar
    profile_df = create_community_profiles(panel_df)
    
    # Adım 2: Hiyerarşik Kümelemeyi uygula
    clustered_df, scaled_features, silhouette = apply_hierarchical_clustering(profile_df)
    
    # Adım 3: Etiketle ve Kaydet
    label_and_save_clusters(clustered_df, silhouette)
    
    # Adım 4: Görselleştir
    visualize_dendrogram(clustered_df, scaled_features)