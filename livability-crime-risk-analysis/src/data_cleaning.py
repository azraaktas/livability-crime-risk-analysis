import pandas as pd
from pathlib import Path


INPUT_PATH = "data/raw/chicago_crimes_2023.csv"
OUTPUT_PATH = "data/processed/chicago_crimes_cleaned.csv"


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def clean_data(df):

    print("İlk veri boyutu:", df.shape)

    # Eksik değer kontrolü
    print("\nEksik değerler:")
    print(df.isnull().sum())

    # Gerekli kolonlarda boş olanları sil
    df = df.dropna(subset=[
        "primary_type",
        "date",
        "community_area",
        "latitude",
        "longitude"
    ])

    print("\nEksik veriler temizlendikten sonra:")
    print(df.shape)

    # Tarih dönüşümü
    df["date"] = pd.to_datetime(df["date"])

    # Yeni zaman özellikleri
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour

    # Community area numeric dönüşümü
    df["community_area"] = pd.to_numeric(
        df["community_area"],
        errors="coerce"
    )

    # Latitude / Longitude dönüşümü
    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # Arrest boolean dönüşümü
    df["arrest"] = df["arrest"].astype(str).str.lower()

    df["arrest"] = df["arrest"].map({
        "true": 1,
        "false": 0
    })

    # Tekrar eksik oluşursa temizle
    df = df.dropna()

    print("\nTemiz veri boyutu:")
    print(df.shape)

    return df


def save_clean_data(df, file_path):

    Path(file_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nTemiz veri kaydedildi: {file_path}")


if __name__ == "__main__":

    df = load_data(INPUT_PATH)

    cleaned_df = clean_data(df)

    print("\nİlk 5 satır:")
    print(cleaned_df.head())

    save_clean_data(cleaned_df, OUTPUT_PATH)