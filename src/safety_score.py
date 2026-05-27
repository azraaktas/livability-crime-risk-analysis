import pandas as pd
from pathlib import Path


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
    return pd.read_csv(file_path)


def add_crime_weights(df):
    df["crime_weight"] = df["primary_type"].map(CRIME_WEIGHTS)

    # Sözlükte olmayan suç türlerine varsayılan ağırlık veriyoruz
    df["crime_weight"] = df["crime_weight"].fillna(3)

    return df


def calculate_safety_score(df):
    grouped_df = df.groupby(
        ["community_area", "year", "month"]
    ).agg(
        total_crime_count=("id", "count"),
        total_crime_weight=("crime_weight", "sum"),
        avg_latitude=("latitude", "mean"),
        avg_longitude=("longitude", "mean")
    ).reset_index()

    min_weight = grouped_df["total_crime_weight"].min()
    max_weight = grouped_df["total_crime_weight"].max()

    grouped_df["risk_score"] = (
        (grouped_df["total_crime_weight"] - min_weight)
        / (max_weight - min_weight)
    ) * 100

    grouped_df["safety_score"] = 100 - grouped_df["risk_score"]

    grouped_df["safety_score"] = grouped_df["safety_score"].round(2)
    grouped_df["risk_score"] = grouped_df["risk_score"].round(2)

    return grouped_df


def save_data(df, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"Güvenlik skoru verisi kaydedildi: {file_path}")


if __name__ == "__main__":
    df = load_data(INPUT_PATH)

    df = add_crime_weights(df)

    safety_df = calculate_safety_score(df)

    print("Güvenlik skoru veri boyutu:", safety_df.shape)
    print(safety_df.head())

    save_data(safety_df, OUTPUT_PATH)