import pandas as pd
from pathlib import Path
import numpy as np

INPUT_PATH = "data/processed/chicago_crimes_cleaned.csv"
OUTPUT_PATH = "data/processed/community_safety_scores.csv"

CRIME_WEIGHTS = {
    "HOMICIDE": 10,
    "CRIMINAL SEXUAL ASSAULT": 9,
    "ROBBERY": 8,
    "ASSAULT": 7,
    "BATTERY": 7,
    "WEAPONS VIOLATION": 7,
    "BURGLARY": 6,
    "MOTOR VEHICLE THEFT": 6,
    "ARSON": 6,
    "THEFT": 4,
    "CRIMINAL DAMAGE": 4,
    "NARCOTICS": 4,
    "DECEPTIVE PRACTICE": 3,
    "OTHER OFFENSE": 2,
    "PUBLIC PEACE VIOLATION": 2
}

def load_data(file_path):
    print(f"Veri yükleniyor: {file_path}")
    return pd.read_csv(file_path)

def add_crime_weights(df):
    df["crime_weight"] = df["primary_type"].map(CRIME_WEIGHTS)
    # Sözlükte olmayan suç türlerine varsayılan ağırlık veriyoruz
    df["crime_weight"] = df["crime_weight"].fillna(3)
    return df

def calculate_safety_score(df):
    # 1. Her bölgenin sabit merkez koordinatlarını çıkaralım (Boş ayları doldururken lazım olacak)
    community_coords = df.groupby("community_area")[["latitude", "longitude"]].mean().reset_index()
    community_coords.rename(columns={"latitude": "avg_latitude", "longitude": "avg_longitude"}, inplace=True)

    # 2. Önce veride var olan ayları gruplayalım ('id' silindiği için 'primary_type' sayıyoruz)
    grouped_df = df.groupby(
        ["community_area", "year_month"]
    ).agg(
        total_crime_count=("primary_type", "count"),
        total_crime_weight=("crime_weight", "sum")
    ).reset_index()

    # 3. ZAMAN SERİSİ İÇİN SIFIRLA DOLDURMA (Zero-Filling / Grid)
    # Bütün mahalleleri (77 bölge) ve verideki bütün ayları (örn. 60 ay) içeren eksiksiz bir ızgara oluşturuyoruz.
    all_communities = df["community_area"].dropna().unique()
    
    min_ym = df["year_month"].min()
    max_ym = df["year_month"].max()
    # Başlangıç ve bitiş arasındaki tüm ayları üret
    all_months = pd.period_range(start=min_ym, end=max_ym, freq='M').astype(str).tolist()

    # Tüm Mahalleler x Tüm Aylar kombinasyonunu (Cross Join) yarat
    grid = pd.MultiIndex.from_product(
        [all_communities, all_months], 
        names=["community_area", "year_month"]
    ).to_frame(index=False)

    # 4. Gerçek verimizle bu tam ızgarayı birleştiriyoruz (Left Join)
    panel_df = pd.merge(grid, grouped_df, on=["community_area", "year_month"], how="left")

    # 5. Suç olmayan ayları 0 ile doldur
    panel_df["total_crime_count"] = panel_df["total_crime_count"].fillna(0)
    panel_df["total_crime_weight"] = panel_df["total_crime_weight"].fillna(0)

    # 6. Koordinatları geri ekle
    panel_df = pd.merge(panel_df, community_coords, on="community_area", how="left")

    # 7. Year ve Month sütunlarını tekrar çıkar (Diğer modellerin ve dashboard'un bozulmaması için)
    panel_df['year'] = panel_df['year_month'].str.split('-').str[0].astype(int)
    panel_df['month'] = panel_df['year_month'].str.split('-').str[1].astype(int)

    # 8. Skorlama Formülü (Senin orijinal formülünün aynısı)
    # Not: Veride sıfırlar olduğu için min_weight doğal olarak 0 veya 0'a çok yakın olacaktır.
    min_weight = panel_df["total_crime_weight"].min()
    max_weight = panel_df["total_crime_weight"].max()

    panel_df["risk_score"] = (
        (panel_df["total_crime_weight"] - min_weight)
        / (max_weight - min_weight)
    ) * 100

    panel_df["safety_score"] = 100 - panel_df["risk_score"]

    panel_df["safety_score"] = panel_df["safety_score"].round(2)
    panel_df["risk_score"] = panel_df["risk_score"].round(2)

    # LSTM'in sırasını bozmamak için veriyi kronolojik ve mahalleye göre sırala
    panel_df = panel_df.sort_values(by=["community_area", "year_month"]).reset_index(drop=True)

    return panel_df

def save_data(df, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"\nGüvenlik skoru (Panel Veri) kaydedildi: {file_path}")

if __name__ == "__main__":
    df = load_data(INPUT_PATH)

    df = add_crime_weights(df)

    safety_df = calculate_safety_score(df)

    print("\nYeni Güvenlik Skoru Veri Boyutu (Panel):", safety_df.shape)
    print("\nİlk 5 Satır:")
    print(safety_df.head())

    save_data(safety_df, OUTPUT_PATH)