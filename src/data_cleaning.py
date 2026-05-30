import pandas as pd
from pathlib import Path

# Dosya yollarını 5 yıllık veriye göre güncelledik
INPUT_PATH = "data/raw/chicago_crimes_2020_2024.csv"
OUTPUT_PATH = "data/processed/chicago_crimes_cleaned.csv"

def load_data(file_path):
    print(f"{file_path} yükleniyor, lütfen bekleyin...")
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    print("İlk veri boyutu:", df.shape)

    # 1. API Çekimlerinden Kaynaklanabilecek Tekrarları Sil
    df = df.drop_duplicates(subset=["id"])
    print("Tekrarlı kayıtlar silindikten sonra:", df.shape)

    # Gerekli kolonlarda boş olanları sil
    df = df.dropna(subset=[
        "primary_type",
        "date",
        "community_area",
        "latitude",
        "longitude"
    ])

    # 2. Tarih Dönüşümü ve Kronolojik Sıralama (Zaman Serisi İçin Kritik)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]) # Hatalı tarih formatlarını uçur
    df = df.sort_values(by="date").reset_index(drop=True) # Eskiden yeniye sırala

    # 3. Yeni Zaman Özellikleri ve LSTM Gruplama Sütunu
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    
    # LSTM'de veriyi aylık gruplamak için YYYY-MM formatında periyot ekliyoruz
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # 4. Bellek Optimizasyonu (Milyonlarca satır için RAM tasarrufu)
    df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce").astype('Int64')
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").astype('float32')
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").astype('float32')

    print("\nModelde kullanılmayacak sütunlar (description, arrest, id) siliniyor...")
    df = df.drop(columns=["description", "arrest", "id"], errors="ignore")

    # İşlemler sonrası tekrar eksik oluşursa temizle
    df = df.dropna()

    print("\nTemiz veri boyutu:")
    print(df.shape)

    return df


def save_clean_data(df, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"\nTemiz veri kaydedildi: {file_path}")

if __name__ == "__main__":
    df = load_data(INPUT_PATH)
    cleaned_df = clean_data(df)

    print("\nİlk 5 satır:")
    print(cleaned_df.head())

    

    save_clean_data(cleaned_df, OUTPUT_PATH)

