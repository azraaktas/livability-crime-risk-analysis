import pandas as pd
import folium
from pathlib import Path


INPUT_PATH = "data/processed/community_clusters.csv"
OUTPUT_PATH = "outputs/chicago_safety_map.html"


def get_color(risk_level):
    if risk_level == "Safe":
        return "green"
    elif risk_level == "Medium Risk":
        return "orange"
    else:
        return "red"


df = pd.read_csv(INPUT_PATH)

chicago_map = folium.Map(
    location=[41.8781, -87.6298],
    zoom_start=11
)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["avg_latitude"], row["avg_longitude"]],
        radius=8,
        color=get_color(row["risk_level"]),
        fill=True,
        fill_color=get_color(row["risk_level"]),
        fill_opacity=0.7,
        popup=(
            f"Community Area: {row['community_name']}<br>"
            f"Safety Score: {row['safety_score']}<br>"
            f"Risk Level: {row['risk_level']}<br>"
            f"Crime Count: {row['total_crime_count']}"
        )
    ).add_to(chicago_map)

Path("outputs").mkdir(exist_ok=True)

chicago_map.save(OUTPUT_PATH)

print(f"Harita oluşturuldu: {OUTPUT_PATH}")