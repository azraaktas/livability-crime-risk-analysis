import streamlit as st
import pandas as pd
import folium
import plotly.express as px

from folium.plugins import HeatMap
from streamlit.components.v1 import html


DATA_PATH = "data/processed/community_clusters.csv"


def get_color(risk_level):
    if risk_level == "Safe":
        return "green"
    elif risk_level == "Medium Risk":
        return "orange"
    return "red"


st.set_page_config(
    page_title="Crime Risk Analysis Dashboard",
    layout="wide"
)

st.title("Yaşanabilirlik Odaklı Güvenlik Skoru ve Suç Analiz Sistemi")

df = pd.read_csv(DATA_PATH)

# SIDEBAR
st.sidebar.header("Filtreleme")

community_list = sorted(df["community_area"].unique())

selected_area = st.sidebar.selectbox(
    "Community Area Seçiniz",
    community_list
)

risk_filter = st.sidebar.multiselect(
    "Risk Seviyesi",
    options=df["risk_level"].unique(),
    default=df["risk_level"].unique()
)

min_score = st.sidebar.slider(
    "Minimum Safety Score",
    0,
    100,
    50
)

filtered_df = df[
    (df["risk_level"].isin(risk_filter)) &
    (df["safety_score"] >= min_score)
]

selected_data = df[
    df["community_area"] == selected_area
].iloc[0]

# KPI CARDS
st.subheader("Bölge Bilgileri")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Safety Score",
    f"{selected_data['safety_score']}"
)

col2.metric(
    "Risk Level",
    selected_data["risk_level"]
)

col3.metric(
    "Crime Count",
    int(selected_data["total_crime_count"])
)

# CHARTS
st.subheader("Risk Dağılımı")

risk_chart = px.histogram(
    filtered_df,
    x="risk_level",
    color="risk_level"
)

st.plotly_chart(risk_chart, use_container_width=True)

st.subheader("Safety Score Dağılımı")

score_chart = px.scatter(
    filtered_df,
    x="total_crime_weight",
    y="safety_score",
    color="risk_level",
    hover_data=["community_area"]
)

st.plotly_chart(score_chart, use_container_width=True)

# MAP
st.subheader("Chicago Risk Haritası")

m = folium.Map(
    location=[41.8781, -87.6298],
    zoom_start=11
)

for _, row in filtered_df.iterrows():

    folium.CircleMarker(
        location=[row["avg_latitude"], row["avg_longitude"]],
        radius=8,
        color=get_color(row["risk_level"]),
        fill=True,
        fill_color=get_color(row["risk_level"]),
        fill_opacity=0.7,
        popup=(
            f"Community Area: {row['community_area']}<br>"
            f"Safety Score: {row['safety_score']}<br>"
            f"Risk Level: {row['risk_level']}"
        )
    ).add_to(m)

heat_data = []

for _, row in filtered_df.iterrows():

    heat_data.append([
        row["avg_latitude"],
        row["avg_longitude"],
        row["total_crime_weight"]
    ])

HeatMap(
    heat_data,
    radius=25,
    blur=15
).add_to(m)

html(m._repr_html_(), height=600)

# MODEL COMPARISON
st.subheader("Model Performans Karşılaştırması")

model_df = pd.DataFrame({
    "Model": [
        "Random Forest",
        "ANN / MLP"
    ],
    "MAE": [
        0.81,
        0.80
    ],
    "R2 Score": [
        0.99,
        0.99
    ]
})

st.dataframe(model_df)

comparison_chart = px.bar(
    model_df,
    x="Model",
    y="R2 Score",
    color="Model"
)

st.plotly_chart(comparison_chart, use_container_width=True)