import pandas as pd
import folium
from folium.plugins import HeatMap
from pathlib import Path

# 77 mahallenin 5 yıllık ortalama risk profilini okuyoruz
INPUT_PATH = "data/processed/community_clusters.csv"
OUTPUT_PATH = "outputs/chicago_crime_heatmap.html"

def load_data():
    print(f"HeatMap verisi yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def create_heatmap(df):
    print("Chicago HeatMap (Isı Haritası) oluşturuluyor...")
    
    # Haritanın arkaplanını koyu tema yapmak (dark_matter) ısı haritasının renklerini (kırmızı/sarı) çok daha iyi gösterir
    chicago_map = folium.Map(
        location=[41.8781, -87.6298],
        zoom_start=11,
        tiles="CartoDB dark_matter" 
    )

    heat_data = []

    # Her bir mahallenin merkez koordinatına, o mahallenin ortalama suç ağırlığını ekliyoruz
    for _, row in df.iterrows():
        heat_data.append([
            row["avg_latitude"],
            row["avg_longitude"],
            row["total_crime_weight"]
        ])

    # Sadece 77 nokta olduğu için radius ve blur değerlerini biraz artırdık ki haritayı güzelce kaplasın
    HeatMap(
        heat_data,
        radius=35, 
        blur=20,
        max_zoom=1
    ).add_to(chicago_map)

    return chicago_map

if __name__ == "__main__":
    df = load_data()
    
    crime_map = create_heatmap(df)

    Path("outputs").mkdir(exist_ok=True)
    crime_map.save(OUTPUT_PATH)

    print(f"\nŞık HeatMap başarıyla oluşturuldu ve kaydedildi: {OUTPUT_PATH}")