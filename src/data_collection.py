import requests
import pandas as pd
from pathlib import Path


API_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"


def fetch_crime_data(limit=5000, start_date="2023-01-01", end_date="2023-12-31"):
    params = {
        "$limit": limit,
        "$select": "id,date,primary_type,description,arrest,community_area,latitude,longitude",
        "$where": f"date between '{start_date}T00:00:00' and '{end_date}T23:59:59'"
    }

    response = requests.get(API_URL, params=params)

    if response.status_code != 200:
        print("Veri çekme başarısız!")
        print("Status Code:", response.status_code)
        print(response.text)
        return None

    data = response.json()
    df = pd.DataFrame(data)

    return df


def save_raw_data(df, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"Veri kaydedildi: {file_path}")


if __name__ == "__main__":
    df = fetch_crime_data()

    if df is not None:
        print("Veri başarıyla çekildi.")
        print("Veri boyutu:", df.shape)
        print(df.head())

        save_raw_data(df, "data/raw/chicago_crimes_2023.csv")