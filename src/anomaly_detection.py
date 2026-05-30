import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Hedef dosyayı 4620 satırlık aylık zaman serisi (panel veri) olarak değiştirdik
INPUT_PATH = "data/processed/community_safety_scores.csv"
OUTPUT_PATH = "outputs/anomaly_results.csv"

def load_data():
    print(f"Anomali tespiti için veri yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def detect_anomalies(df):
    print("Isolation Forest modeli ile zaman serisi anomalileri aranıyor...")
    
    # Anomali aranacak özellikler
    features = df[[
        "total_crime_count",
        "total_crime_weight",
        "safety_score"
    ]]

    # Veriyi ölçeklendirme (Standartlaştırma)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Isolation Forest Modeli
    # contamination=0.05 demek: "Verilerin en tuhaf %5'ini anomali olarak işaretle"
    model = IsolationForest(
        contamination=0.05, 
        random_state=42,
        n_jobs=-1
    )

    df["anomaly_label"] = model.fit_predict(scaled_features)

    df["anomaly_status"] = df["anomaly_label"].map({
        1: "Normal",
        -1: "Anomaly"
    })
    
    # Kaç tane anomali bulunduğunu raporla
    anomaly_count = df[df["anomaly_status"] == "Anomaly"].shape[0]
    print(f"Toplam {df.shape[0]} aylık veri içinde {anomaly_count} adet anomali (ani suç patlaması veya aşırı düşüş) tespit edildi.")

    return df

def save_data(df, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"\nAnomali sonuçları kaydedildi: {file_path}")

if __name__ == "__main__":
    df = load_data()
    
    df_anomalies = detect_anomalies(df)

    print("\nAnomali Tespit Edilen İlk 10 Örnek (Zaman ve Bölge Bazlı):")
    print("---------------------------------------------------------")
    
    # Sadece anomalileri filtreleyip ekrana yazdırıyoruz ki modelin neyi yakaladığını görelim
    anomalies_only = df_anomalies[df_anomalies["anomaly_status"] == "Anomaly"]
    
    print(anomalies_only[[
        "community_area",
        "year_month", # Artık HANGİ AYDA anomali yaşandığını görebiliyoruz
        "safety_score",
        "total_crime_weight",
        "anomaly_status"
    ]].head(10))

    save_data(df_anomalies, OUTPUT_PATH)