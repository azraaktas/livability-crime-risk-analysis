import pandas as pd
import folium
from folium.plugins import HeatMap
from pathlib import Path


INPUT_PATH = "data/processed/community_clusters.csv"
OUTPUT_PATH = "outputs/chicago_crime_heatmap.html"


df = pd.read_csv(INPUT_PATH)

chicago_map = folium.Map(
    location=[41.8781, -87.6298],
    zoom_start=11
)

heat_data = []

for _, row in df.iterrows():
    heat_data.append([
        row["avg_latitude"],
        row["avg_longitude"],
        row["total_crime_weight"]
    ])

HeatMap(
    heat_data,
    radius=25,
    blur=15
).add_to(chicago_map)

Path("outputs").mkdir(exist_ok=True)

chicago_map.save(OUTPUT_PATH)

print(f"HeatMap oluşturuldu: {OUTPUT_PATH}")