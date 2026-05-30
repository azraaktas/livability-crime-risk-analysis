import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

API_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"


def fetch_monthly_data(start_date="2015-01-01", end_date="2025-12-31", limit=50000):
    all_data = []

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        month_start = current
        month_end = current + relativedelta(months=1) - relativedelta(days=1)

        print(f"Veri çekiliyor: {month_start.strftime('%Y-%m')}")

        params = {
            "$limit": limit,
            "$select": "id,date,primary_type,description,arrest,community_area,latitude,longitude",
            "$where": f"date between '{month_start.strftime('%Y-%m-%d')}T00:00:00' and '{month_end.strftime('%Y-%m-%d')}T23:59:59'",
            "$order": "date DESC"
        }

        response = requests.get(API_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            all_data.extend(data)
            print(f"{len(data)} kayıt çekildi.")
        else:
            print("Hata:", response.status_code)
            print(response.text)

        current = current + relativedelta(months=1)

    df = pd.DataFrame(all_data)
    return df


def save_raw_data(df, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"Veri kaydedildi: {file_path}")


if __name__ == "__main__":
    df = fetch_monthly_data()

    print("Toplam veri boyutu:", df.shape)
    print(df.head())

    save_raw_data(df, "data/raw/chicago_crimes_2015_2025_monthly.csv")