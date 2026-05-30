import requests
import pandas as pd
from pathlib import Path
import time

API_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

def fetch_crime_data(start_date="2020-01-01", end_date="2024-12-31", chunk_size=50000):
    all_data = []
    offset = 0

    print(f"{start_date} ile {end_date} arası veri çekiliyor. Bu işlem birkaç dakika sürebilir...")

    while True:
        params = {
            "$limit": chunk_size,
            "$offset": offset,
            "$select": "id,date,primary_type,description,arrest,community_area,latitude,longitude",
            "$where": f"date between '{start_date}T00:00:00' and '{end_date}T23:59:59'",
            "$order": "date ASC" # Veriyi eskiden yeniye kronolojik olarak sırala
        }

        response = requests.get(API_URL, params=params)

        if response.status_code != 200:
            print("Veri çekme başarısız!")
            print("Status Code:", response.status_code)
            print(response.text)
            break

        data = response.json()
        
        # Eğer API'den boş liste dönerse, çekilecek veri kalmamıştır
        if not data:
            break 

        all_data.extend(data)
        print(f"{len(all_data)} satır veri çekildi...")

        # Eğer gelen veri chunk_size'dan küçükse, son partiyi almışız demektir
        if len(data) < chunk_size:
            break 

        offset += chunk_size
        time.sleep(1) # Socrata API'sini yorup ban yememek için 1 saniye bekleme

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    return df

def save_raw_data(df, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"Veri başarıyla kaydedildi: {file_path}")

if __name__ == "__main__":
    # LSTM için daha geniş bir zaman aralığı (5 yıl) belirliyoruz
    df = fetch_crime_data(start_date="2020-01-01", end_date="2024-12-31")

    if df is not None:
        print("\nVeri çekme işlemi tamamlandı.")
        print("Toplam Veri Boyutu:", df.shape)
        print(df.head())

        # Dosya adını yılları kapsayacak şekilde güncelledik
        save_raw_data(df, "data/raw/chicago_crimes_2020_2024.csv")