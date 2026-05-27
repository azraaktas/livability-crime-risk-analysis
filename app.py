import streamlit as st
import pandas as pd
import folium
import plotly.express as px

from PIL import Image
from folium.plugins import HeatMap
from streamlit.components.v1 import html


DATA_PATH = "data/processed/community_clusters.csv"
MODEL_PATH = "outputs/model_comparison_results.csv"
CLUSTER_METRICS_PATH = "outputs/clustering_metrics.csv"
PREDICTIONS_PATH = "outputs/model_predictions.csv"

HIERARCHICAL_DATA_PATH = "data/processed/hierarchical_clusters.csv"
HIERARCHICAL_METRICS_PATH = "outputs/hierarchical_clustering_metrics.csv"
DENDROGRAM_PATH = "outputs/hierarchical_dendrogram.png"
FEATURE_IMPORTANCE_PATH = "outputs/feature_importance.csv"
ANOMALY_PATH = "outputs/anomaly_results.csv"


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
anomaly_df = pd.read_csv(ANOMALY_PATH)
feature_importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

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

selected_anomaly = anomaly_df[
    anomaly_df["community_area"] == selected_area
].iloc[0]

anomaly_status = selected_anomaly["anomaly_status"]

# KPI CARDS
st.subheader("Bölge Bilgileri")

col1, col2, col3 = st.columns(3)

col1.metric("Safety Score", f"{selected_data['safety_score']}")
col2.metric("Risk Level", selected_data["risk_level"])
col3.metric("Crime Count", int(selected_data["total_crime_count"]))

# AUTOMATIC RISK INTERPRETATION
st.subheader("Otomatik Risk Yorumu")

score = selected_data["safety_score"]
risk = selected_data["risk_level"]
crime_count = selected_data["total_crime_count"]
crime_weight = selected_data["total_crime_weight"]

if risk == "Safe":
    comment = (
        f"Seçilen bölgenin güvenlik skoru {score} olarak hesaplanmıştır. "
        f"Bu değer bölgenin düşük riskli ve daha güvenli bir kategoriye ait olduğunu göstermektedir. "
        
    )
elif risk == "Medium Risk":
    comment = (
        f"Seçilen bölgenin güvenlik skoru {score} olarak hesaplanmıştır. "
        f"Bölge orta risk seviyesindedir. "
        f"Bu bölgede suç yoğunluğu tamamen düşük değildir, bu nedenle düzenli izleme önerilir. "
        
    )
else:
    comment = (
        f"Seçilen bölgenin güvenlik skoru {score} olarak hesaplanmıştır. "
        f"Bu bölge yüksek risk kategorisindedir. "
        f"Suç ağırlığı ve suç yoğunluğu diğer bölgelere göre daha fazladır. "
        f"Bu nedenle bölge güvenlik açısından öncelikli incelenmelidir. "
        
    )

st.info(comment)

# DATA-DRIVEN RECOMMENDATION ENGINE
st.subheader("Veri Odaklı Risk Öneri Sistemi")

recommendations = []

avg_crime_weight = df["total_crime_weight"].mean()
avg_crime_count = df["total_crime_count"].mean()

if score < 40:
    recommendations.append(
        "Güvenlik skoru oldukça düşük olduğu için bölge öncelikli risk alanı olarak değerlendirilmelidir."
    )
elif score < 70:
    recommendations.append(
        "Güvenlik skoru orta seviyededir; bölge düzenli olarak takip edilmelidir."
    )
else:
    recommendations.append(
        "Güvenlik skoru yüksek olduğu için mevcut güvenlik seviyesi korunmalıdır."
    )

if crime_weight > avg_crime_weight:
    recommendations.append(
        "Toplam suç ağırlığı ortalamanın üzerindedir; ağır suç türlerinin yoğunluğu ayrıca incelenmelidir."
    )

if crime_count > avg_crime_count:
    recommendations.append(
        "Toplam suç sayısı ortalamanın üzerindedir; suç yoğunluğunun arttığı bölgeler detaylandırılmalıdır."
    )

if anomaly_status == "Anomaly":
    recommendations.append(
        "Isolation Forest modeline göre bu bölge diğer bölgelerden istatistiksel olarak farklı davranmaktadır; anormal risk analizi yapılmalıdır."
    )

top_feature_df = feature_importance_df[
    feature_importance_df["Model"] == "Random Forest"
].sort_values(
    by="Importance",
    ascending=False
)



for rec in recommendations:
    st.warning(rec)

# TOP RISK TABLES
st.subheader("En Güvenli ve En Riskli Bölgeler")

top_safe = df.sort_values(
    by="safety_score",
    ascending=False
).head(10)

top_risky = df.sort_values(
    by="safety_score",
    ascending=True
).head(10)

safe_col, risky_col = st.columns(2)

with safe_col:
    st.markdown("### En Güvenli 10 Bölge")
    st.dataframe(
        top_safe[[
            "community_area",
            "safety_score",
            "risk_level",
            "total_crime_count",
            "total_crime_weight"
        ]],
        use_container_width=True
    )

with risky_col:
    st.markdown("### En Riskli 10 Bölge")
    st.dataframe(
        top_risky[[
            "community_area",
            "safety_score",
            "risk_level",
            "total_crime_count",
            "total_crime_weight"
        ]],
        use_container_width=True
    )

# AREA COMPARISON
st.subheader("Bölge Karşılaştırma")

area_col1, area_col2 = st.columns(2)

with area_col1:
    compare_area_1 = st.selectbox(
        "1. Bölgeyi Seçiniz",
        community_list,
        key="compare_area_1"
    )

with area_col2:
    compare_area_2 = st.selectbox(
        "2. Bölgeyi Seçiniz",
        community_list,
        key="compare_area_2"
    )

area_1_data = df[df["community_area"] == compare_area_1].iloc[0]
area_2_data = df[df["community_area"] == compare_area_2].iloc[0]

comparison_df = pd.DataFrame({
    "Metric": [
        "Safety Score",
        "Risk Level",
        "Total Crime Count",
        "Total Crime Weight"
    ],
    f"Community Area {compare_area_1}": [
        area_1_data["safety_score"],
        area_1_data["risk_level"],
        area_1_data["total_crime_count"],
        area_1_data["total_crime_weight"]
    ],
    f"Community Area {compare_area_2}": [
        area_2_data["safety_score"],
        area_2_data["risk_level"],
        area_2_data["total_crime_count"],
        area_2_data["total_crime_weight"]
    ]
})

st.dataframe(
    comparison_df,
    use_container_width=True
)

comparison_bar_df = pd.DataFrame({
    "Community Area": [
        f"Area {compare_area_1}",
        f"Area {compare_area_2}"
    ],
    "Safety Score": [
        area_1_data["safety_score"],
        area_2_data["safety_score"]
    ],
    "Total Crime Weight": [
        area_1_data["total_crime_weight"],
        area_2_data["total_crime_weight"]
    ]
})

comparison_score_chart = px.bar(
    comparison_bar_df,
    x="Community Area",
    y="Safety Score",
    text="Safety Score",
    title="Seçilen Bölgelerin Safety Score Karşılaştırması"
)

comparison_score_chart.update_traces(textposition="outside")

st.plotly_chart(
    comparison_score_chart,
    use_container_width=True
)

# ANOMALY DETECTION
st.subheader("Anormal Risk Analizi")

anomaly_chart = px.scatter(
    anomaly_df,
    x="total_crime_weight",
    y="safety_score",
    color="anomaly_status",
    hover_data=[
        "community_area",
        "risk_level",
        "total_crime_count"
    ],
    title="Isolation Forest Anomaly Detection"
)

st.plotly_chart(
    anomaly_chart,
    use_container_width=True
)

anomalies_only = anomaly_df[
    anomaly_df["anomaly_status"] == "Anomaly"
]

st.markdown("### Tespit Edilen Anormal Bölgeler")

st.dataframe(
    anomalies_only[[
        "community_area",
        "safety_score",
        "risk_level",
        "total_crime_count",
        "total_crime_weight"
    ]],
    use_container_width=True
)

# K-MEANS CLUSTERING RESULTS
st.subheader("K-Means Kümeleme Sonuçları")

cluster_metrics_df = pd.read_csv(CLUSTER_METRICS_PATH)

silhouette_value = cluster_metrics_df.loc[
    cluster_metrics_df["Metric"] == "Silhouette Score",
    "Value"
].values[0]

st.metric("K-Means Silhouette Score", silhouette_value)

cluster_chart = px.scatter(
    filtered_df,
    x="total_crime_weight",
    y="safety_score",
    color="risk_level",
    hover_data=[
        "community_area",
        "cluster",
        "total_crime_count"
    ],
    title="K-Means Cluster Grafiği"
)

st.plotly_chart(cluster_chart, use_container_width=True)

# HIERARCHICAL CLUSTERING RESULTS
st.subheader("Hierarchical Clustering Sonuçları")

hierarchical_df = pd.read_csv(HIERARCHICAL_DATA_PATH)
hierarchical_metrics_df = pd.read_csv(HIERARCHICAL_METRICS_PATH)

hierarchical_silhouette = hierarchical_metrics_df.loc[
    hierarchical_metrics_df["Metric"] == "Silhouette Score",
    "Value"
].values[0]

st.metric("Hierarchical Silhouette Score", hierarchical_silhouette)

hierarchical_filtered_df = hierarchical_df[
    (hierarchical_df["hierarchical_risk_level"].isin(risk_filter)) &
    (hierarchical_df["safety_score"] >= min_score)
]

hierarchical_chart = px.scatter(
    hierarchical_filtered_df,
    x="total_crime_weight",
    y="safety_score",
    color="hierarchical_risk_level",
    hover_data=[
        "community_area",
        "hierarchical_cluster",
        "total_crime_count"
    ],
    title="Hierarchical Clustering Grafiği"
)

st.plotly_chart(hierarchical_chart, use_container_width=True)

st.subheader("Hierarchical Clustering Dendrogram")

dendrogram_image = Image.open(DENDROGRAM_PATH)

st.image(
    dendrogram_image,
    caption="Community Area bazlı Hierarchical Clustering Dendrogram",
    use_container_width=True
)

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
            f"Risk Level: {row['risk_level']}<br>"
            f"Crime Count: {row['total_crime_count']}"
        )
    ).add_to(m)

heat_data = []

for _, row in filtered_df.iterrows():
    heat_data.append([
        row["avg_latitude"],
        row["avg_longitude"],
        row["total_crime_weight"]
    ])

if len(heat_data) > 0:
    HeatMap(
        heat_data,
        radius=25,
        blur=15
    ).add_to(m)

html(m._repr_html_(), height=600)

# MODEL COMPARISON
st.subheader("Model Performans Karşılaştırması")

model_df = pd.read_csv(MODEL_PATH)

st.dataframe(model_df, use_container_width=True)

comparison_chart = px.bar(
    model_df,
    x="Model",
    y="R2 Score",
    color="Model",
    text="R2 Score",
    title="Modellere Göre R2 Score Karşılaştırması"
)

comparison_chart.update_traces(textposition="outside")

st.plotly_chart(comparison_chart, use_container_width=True)

error_chart = px.bar(
    model_df,
    x="Model",
    y="RMSE",
    color="Model",
    text="RMSE",
    title="Modellere Göre RMSE Karşılaştırması"
)

error_chart.update_traces(textposition="outside")

st.plotly_chart(error_chart, use_container_width=True)

# FEATURE IMPORTANCE
st.subheader("Feature Importance Analizi")

selected_importance_model = st.selectbox(
    "Feature importance için model seçiniz:",
    feature_importance_df["Model"].unique()
)

selected_importance_df = feature_importance_df[
    feature_importance_df["Model"] == selected_importance_model
]

importance_chart = px.bar(
    selected_importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Feature",
    text="Importance",
    title=f"{selected_importance_model} Feature Importance"
)

importance_chart.update_traces(textposition="outside")

st.plotly_chart(importance_chart, use_container_width=True)

st.dataframe(selected_importance_df, use_container_width=True)

# MODEL PREDICTIONS
st.subheader("Gerçek ve Tahmin Edilen Safety Score Karşılaştırması")

predictions_df = pd.read_csv(PREDICTIONS_PATH)

selected_model = st.selectbox(
    "Tahmin grafiği için model seçiniz:",
    predictions_df["Model"].unique()
)

selected_predictions = predictions_df[
    predictions_df["Model"] == selected_model
]

prediction_chart = px.scatter(
    selected_predictions,
    x="Actual Safety Score",
    y="Predicted Safety Score",
    title=f"{selected_model} - Gerçek vs Tahmin Safety Score",
    labels={
        "Actual Safety Score": "Gerçek Safety Score",
        "Predicted Safety Score": "Tahmin Edilen Safety Score"
    }
)

prediction_chart.add_shape(
    type="line",
    x0=selected_predictions["Actual Safety Score"].min(),
    y0=selected_predictions["Actual Safety Score"].min(),
    x1=selected_predictions["Actual Safety Score"].max(),
    y1=selected_predictions["Actual Safety Score"].max(),
    line=dict(dash="dash")
)

st.plotly_chart(prediction_chart, use_container_width=True)

st.dataframe(selected_predictions, use_container_width=True)