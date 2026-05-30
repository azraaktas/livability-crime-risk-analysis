import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from folium.plugins import HeatMap
from streamlit.components.v1 import html as st_html

# ── MUST BE FIRST STREAMLIT CALL ────────────────────────────────────────────
st.set_page_config(
    page_title="Chicago Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# ── COMMUNITY NAMES (UNCHANGED) ──────────────────────────────────────────────
community_names = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square", 5: "North Center",
    6: "Lake View", 7: "Lincoln Park", 8: "Near North Side", 9: "Edison Park", 10: "Norwood Park",
    11: "Jefferson Park", 12: "Forest Glen", 13: "North Park", 14: "Albany Park", 15: "Portage Park",
    16: "Irving Park", 17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
    21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town", 25: "Austin",
    26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side", 29: "North Lawndale",
    30: "South Lawndale", 31: "Lower West Side", 32: "Loop", 33: "Near South Side", 34: "Armour Square",
    35: "Douglas", 36: "Oakland", 37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood",
    40: "Washington Park", 41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham",
    45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights", 49: "Roseland",
    50: "Pullman", 51: "South Deering", 52: "East Side", 53: "West Pullman", 54: "Riverdale",
    55: "Hegewisch", 56: "Garfield Ridge", 57: "Archer Heights", 58: "Brighton Park", 59: "McKinley Park",
    60: "Bridgeport", 61: "New City", 62: "West Elsdon", 63: "Gage Park", 64: "Clearing",
    65: "West Lawn", 66: "Chicago Lawn", 67: "West Englewood", 68: "Englewood",
    69: "Greater Grand Crossing", 70: "Ashburn", 71: "Auburn Gresham", 72: "Beverly",
    73: "Washington Heights", 74: "Mount Greenwood", 75: "Morgan Park", 76: "O'Hare", 77: "Edgewater"
}

# ── FILE PATHS (UNCHANGED) ───────────────────────────────────────────────────
DATA_PATH              = "data/processed/community_clusters.csv"
MODEL_PATH             = "outputs/model_comparison_results.csv"
CLUSTER_METRICS_PATH   = "outputs/clustering_metrics.csv"
PREDICTIONS_PATH       = "outputs/model_predictions.csv"
HIERARCHICAL_DATA_PATH = "data/processed/hierarchical_clusters.csv"
HIERARCHICAL_METRICS_PATH = "outputs/hierarchical_clustering_metrics.csv"
DENDROGRAM_PATH        = "outputs/hierarchical_dendrogram.png"
FEATURE_IMPORTANCE_PATH= "outputs/feature_importance.csv"
ANOMALY_PATH           = "outputs/anomaly_results.csv"

# ── DESIGN TOKENS ────────────────────────────────────────────────────────────
RISK_HEX = {"Safe": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#F43F5E"}
CHART_PALETTE = ["#00D4FF", "#10B981", "#F59E0B", "#F43F5E", "#A78BFA", "#FB923C", "#34D399"]

# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Barlow+Condensed:wght@500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ─ Reset & Base ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}
.stApp {
    background: #070C18 !important;
}
.main .block-container {
    padding: 1.25rem 2.25rem 4rem !important;
    max-width: 1600px !important;
}

/* ─ Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0B1120 !important;
    border-right: 1px solid rgba(0,212,255,0.08) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ─ Streamlit Metric ─────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0E1528 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    color: #4A5568 !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: #E8EDF5 !important;
    font-size: 1.55rem !important;
}

/* ─ Tabs ─────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: #0E1528 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #4A5568 !important;
    border-radius: 7px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.01em !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: rgba(0,212,255,0.1) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
}

/* ─ Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0097B2, #00D4FF) !important;
    color: #07090F !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.8rem !important;
    letter-spacing: 0.03em !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.25) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.4) !important;
}

/* ─ Selects / Multiselect ────────────────────────────────── */
[data-baseweb="select"] > div {
    background: #111B30 !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #C8D3E0 !important;
    border-radius: 8px !important;
}
[data-baseweb="menu"] { background: #111B30 !important; }
[data-baseweb="option"] { color: #C8D3E0 !important; }
[data-baseweb="option"]:hover { background: rgba(0,212,255,0.08) !important; }
[data-baseweb="tag"] {
    background: rgba(0,212,255,0.12) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
}

/* ─ Slider ───────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background: #00D4FF !important;
    border: 2px solid #00D4FF !important;
}
[data-testid="stSlider"] > div > div > div[style*="background"] {
    background: #00D4FF !important;
}

/* ─ Alerts ───────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ─ Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ─ Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0E1528 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #7A8BA0 !important;
    font-size: 0.83rem !important;
}

/* ─ Custom Components ────────────────────────────────────── */
.dash-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 1rem 0 1.25rem;
    border-bottom: 1px solid rgba(0,212,255,0.1);
    margin-bottom: 1.5rem;
}
.dash-title {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #E8EDF5 !important;
    letter-spacing: 0.04em !important;
    line-height: 1 !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}
.dash-subtitle {
    font-size: 0.8rem !important;
    color: #3A4A60 !important;
    margin: 4px 0 0 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
.city-tag {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #00D4FF;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Section header */
.sec-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 1rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sec-icon {
    width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,212,255,0.08);
    border-radius: 7px;
    font-size: 1rem;
    flex-shrink: 0;
}
.sec-title {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: #7A8BA0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    margin: 0 !important;
}

/* Score card */
.score-card {
    background: linear-gradient(145deg, #0E1528, #111B30);
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 30px rgba(0,0,0,0.5);
    margin: 12px 0 20px;
}
.score-number {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin: 8px 0 4px;
}
.score-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #3A4A60;
}
.score-track {
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 6px;
    margin-top: 14px;
    overflow: hidden;
}

/* Risk badge */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-safe   { background: rgba(16,185,129,0.12); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
.badge-medium { background: rgba(245,158,11,0.12); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
.badge-high   { background: rgba(244,63,94,0.12);  color: #F43F5E; border: 1px solid rgba(244,63,94,0.3);  }

/* Sidebar inner */
.sb-logo {
    padding: 24px 20px 16px;
    border-bottom: 1px solid rgba(0,212,255,0.08);
    margin-bottom: 20px;
}
.sb-section {
    padding: 0 16px;
    margin-bottom: 14px;
}
.sb-label {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #2A3A50;
    margin-bottom: 5px;
    display: block;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 16px;
}
.stat-row-label { font-size: 0.78rem; color: #4A5568; }
.stat-val-safe   { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #10B981; font-weight: 700; }
.stat-val-medium { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #F59E0B; font-weight: 700; }
.stat-val-high   { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #F43F5E; font-weight: 700; }

/* Comparison table */
.cmp-table { width: 100%; border-collapse: separate; border-spacing: 0; border-radius: 10px; overflow: hidden; }
.cmp-th {
    background: #0E1528;
    color: #3A4A60;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 10px 16px;
    text-align: left;
    font-weight: 700;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.cmp-td {
    background: #0B1120;
    color: #C8D3E0;
    padding: 10px 16px;
    font-size: 0.85rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.cmp-td-metric { color: #4A5568; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }

/* Glow divider */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.2), transparent);
    margin: 2rem 0;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ──────────────────────────────────────────────────────────────────
def get_color(risk_level):
    return {"Safe": "green", "Medium Risk": "orange"}.get(risk_level, "red")

def badge_html(risk_level):
    cls = {"Safe": "badge-safe", "Medium Risk": "badge-medium", "High Risk": "badge-high"}.get(risk_level, "badge-high")
    dot = {"Safe": "●", "Medium Risk": "●", "High Risk": "●"}.get(risk_level, "●")
    color = RISK_HEX.get(risk_level, "#888")
    return f'<span class="badge {cls}" style="color:{color};">{dot} {risk_level}</span>'

def score_bar(pct, risk_level):
    color = RISK_HEX.get(risk_level, "#888")
    glow  = {"Safe": "rgba(16,185,129,0.4)", "Medium Risk": "rgba(245,158,11,0.4)", "High Risk": "rgba(244,63,94,0.4)"}.get(risk_level, "rgba(255,255,255,0.2)")
    return f"""
    <div class="score-track">
        <div style="width:{pct}%; height:100%; border-radius:99px;
                    background:{color}; box-shadow: 0 0 10px {glow}; transition: width 0.6s ease;">
        </div>
    </div>"""

def sec(icon, title):
    st.markdown(f"""
    <div class="sec-header">
        <div class="sec-icon">{icon}</div>
        <p class="sec-title">{title}</p>
    </div>""", unsafe_allow_html=True)

def style_chart(fig, title=""):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,17,32,0.95)",
        font=dict(color="#4A5568", family="DM Sans, sans-serif", size=11),
        title=dict(text=title, font=dict(size=13, color="#C8D3E0",
                   family="Barlow Condensed, sans-serif"), x=0, xanchor="left") if title else None,
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)",
                   tickfont=dict(color="#3A4A60"), zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)",
                   tickfont=dict(color="#3A4A60"), zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7A8BA0"), bordercolor="rgba(255,255,255,0.05)"),
        colorway=CHART_PALETTE,
        margin=dict(l=8, r=8, t=44 if title else 12, b=8),
        hoverlabel=dict(bgcolor="#111B30", font_color="#E8EDF5", bordercolor="rgba(0,212,255,0.3)"),
    )
    return fig


# ── DATA LOADING (LOGIC UNCHANGED) ───────────────────────────────────────────
@st.cache_data
def load_all_data():
    df = pd.read_csv(DATA_PATH)
    df["community_area"] = df["community_area"].astype(int)
    df["community_name"]  = df["community_area"].map(community_names)

    anomaly_df = pd.read_csv(ANOMALY_PATH)
    anomaly_df["community_area"] = anomaly_df["community_area"].astype(int)
    anomaly_df["community_name"]  = anomaly_df["community_area"].map(community_names)

    feature_importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    model_df = pd.read_csv(MODEL_PATH)
    if "LSTM (Deep Learning)" not in model_df["Model"].values:
        lstm_row = pd.DataFrame([{
            "Model": "LSTM (Deep Learning)",
            "MAE": 1.9848, "MSE": 7.9933, "RMSE": 2.8272, "R2 Score": 0.9729
        }])
        model_df = pd.concat([model_df, lstm_row], ignore_index=True)
        model_df = model_df.sort_values(by="R2 Score", ascending=False)

    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    try:
        lstm_preds = pd.read_csv("outputs/lstm_predictions.csv")
        predictions_df = pd.concat([predictions_df, lstm_preds], ignore_index=True)
    except FileNotFoundError:
        st.warning("LSTM tahmin dosyası henüz oluşturulmamış.")

    hierarchical_df = pd.read_csv(HIERARCHICAL_DATA_PATH)
    return df, anomaly_df, feature_importance_df, model_df, predictions_df, hierarchical_df

df, anomaly_df, feature_importance_df, model_df, predictions_df, hierarchical_df = load_all_data()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div style="font-size:1.6rem; margin-bottom:6px;">🛡️</div>
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.05rem;
                    font-weight:700; color:#E8EDF5; letter-spacing:0.1em; text-transform:uppercase;">
            Chicago Risk Intel
        </div>
        <div style="font-size:0.65rem; color:#2A3A50; font-weight:700;
                    text-transform:uppercase; letter-spacing:0.16em; margin-top:2px;">
            Crime Analysis Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    community_list = sorted(df["community_name"].unique())

    st.markdown('<div class="sb-section"><span class="sb-label">📍 Bölge Seç</span>', unsafe_allow_html=True)
    selected_area = st.selectbox("", community_list, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><span class="sb-label">⚠️ Risk Filtresi</span>', unsafe_allow_html=True)
    risk_filter = st.multiselect("", options=df["risk_level"].unique(),
                                  default=df["risk_level"].unique(), label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><span class="sb-label">📊 Min. Güvenlik Skoru</span>', unsafe_allow_html=True)
    min_score = st.slider("", 0, 100, 50, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Compute filtered_df here so sidebar stats work
    filtered_df = df[(df["risk_level"].isin(risk_filter)) & (df["safety_score"] >= min_score)]

    n_safe   = len(filtered_df[filtered_df["risk_level"] == "Safe"])
    n_medium = len(filtered_df[filtered_df["risk_level"] == "Medium Risk"])
    n_high   = len(filtered_df[filtered_df["risk_level"] == "High Risk"])

    st.markdown(f"""
    <hr style="border:none; border-top:1px solid rgba(0,212,255,0.08); margin:18px 0 12px;">
    <div style="padding:0 16px; margin-bottom:4px;">
        <span class="sb-label">Aktif Filtre Özeti</span>
    </div>
    <div class="stat-row"><span class="stat-row-label">● Güvenli Bölge</span>     <span class="stat-val-safe">{n_safe}</span></div>
    <div class="stat-row"><span class="stat-row-label">● Orta Riskli</span>       <span class="stat-val-medium">{n_medium}</span></div>
    <div class="stat-row"><span class="stat-row-label">● Yüksek Riskli</span>     <span class="stat-val-high">{n_high}</span></div>
    <div class="stat-row"><span class="stat-row-label" style="color:#3A4A60;">Toplam Bölge</span>
        <span style="font-family:'Space Mono',monospace; font-size:0.85rem; color:#7A8BA0; font-weight:700;">
            {n_safe + n_medium + n_high}</span>
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <div>
        <p class="dash-title">🔐 Güvenlik Skoru & Suç Analiz Sistemi</p>
        <p class="dash-subtitle">Chicago, Illinois &nbsp;·&nbsp; Yaşanabilirlik Odaklı Risk Değerlendirmesi</p>
    </div>
    <div class="city-tag">🌆 &nbsp;Chicago IL</div>
</div>
""", unsafe_allow_html=True)


# ── SELECTED AREA (LOGIC UNCHANGED) ──────────────────────────────────────────
selected_data = df[df["community_name"] == selected_area].iloc[0]
area_anomalies = anomaly_df[
    (anomaly_df["community_name"] == selected_area) &
    (anomaly_df["anomaly_status"] == "Anomaly")
]
anomaly_status = "Anomaly" if len(area_anomalies) > 0 else "Normal"
score        = selected_data["safety_score"]
risk         = selected_data["risk_level"]
crime_count  = selected_data["total_crime_count"]
crime_weight = selected_data["total_crime_weight"]
avg_crime_weight = df["total_crime_weight"].mean()


# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Genel Bakış",
    "🗺️  Harita",
    "🤖  Model Analizi",
    "📊  Kümeleme",
    "🔬  Gelişmiş Analiz",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GENEL BAKIŞ
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI Cards ──
    sec("📌", f"{selected_area} — Bölge Özeti")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Güvenlik Skoru", f"{score:.1f} / 100")
    with c2:
        st.metric("Risk Seviyesi", risk)
    with c3:
        st.metric("Aylık Suç (Ort.)", f"{int(crime_count):,}")
    with c4:
        st.metric("Anomali Durumu",
                  "⚠️ Anomali Var" if anomaly_status == "Anomaly" else "✅ Normal")

    # ── Score Visualiser Card ──
    risk_color  = RISK_HEX.get(risk, "#888")
    glow_shadow = {"Safe": "0 0 40px rgba(16,185,129,0.15)",
                   "Medium Risk": "0 0 40px rgba(245,158,11,0.15)",
                   "High Risk": "0 0 40px rgba(244,63,94,0.15)"}.get(risk, "none")
    bar_html    = score_bar(score, risk)
    bdg         = badge_html(risk)

    st.markdown(f"""
    <div class="score-card" style="border-left:3px solid {risk_color}; box-shadow:{glow_shadow};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
            <span style="font-size:0.85rem; font-weight:600; color:#7A8BA0;">{selected_area}</span>
            {bdg}
        </div>
        <div class="score-number" style="color:{risk_color};">{score:.1f}</div>
        <div class="score-label">Safety Score</div>
        {bar_html}
        <div style="display:flex; justify-content:space-between; margin-top:6px;">
            <span style="font-size:0.7rem; color:#2A3A50;">0 — Tehlikeli</span>
            <span style="font-size:0.7rem; color:#2A3A50;">100 — Güvenli</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risk Comment (LOGIC UNCHANGED) ──
    sec("💡", "Risk Yorumu")
    if risk == "Safe":
        comment = f"Seçilen bölgenin güvenlik skoru **{score:.1f}** olarak hesaplanmıştır. K-Means profillemesine göre bölge **düşük riskli (Safe)** kategoriye aittir. Konut seçimi veya ticari yatırımlar için oldukça uygun bir profil sergilemektedir."
    elif risk == "Medium Risk":
        comment = f"Seçilen bölgenin güvenlik skoru **{score:.1f}** olarak hesaplanmıştır. Bölge **orta risk (Medium Risk)** seviyesindedir. Suç yoğunluğu tamamen düşük değildir, aylık dalgalanmalar yaşanabilmektedir. Düzenli izleme önerilir."
    else:
        comment = f"Seçilen bölgenin güvenlik skoru **{score:.1f}** olarak hesaplanmıştır. Bu bölge **yüksek risk (High Risk)** kategorisindedir! Suç ağırlığı ve yoğunluğu diğer bölgelere göre tehlikeli boyuttadır. Güvenlik açısından öncelikli incelenmelidir."
    st.info(comment)

    # ── Recommendations (LOGIC UNCHANGED) ──
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    sec("🛠️", "Tavsiyeler & Uyarılar")

    recommendations = []
    if score < 40:
        recommendations.append(("error",   "Güvenlik skoru kritik eşiğin altındadır, acil kolluk kuvveti takviyesi gereklidir."))
    elif score < 70:
        recommendations.append(("warning", "Orta seviye güvenlik skoru. Özellikle gece saatlerindeki suç trendleri incelenmelidir."))
    if crime_weight > avg_crime_weight:
        recommendations.append(("warning", "Toplam suç ağırlığı şehir ortalamasının üzerindedir. Ağır suç türlerine (şiddet/soygun) yatkınlık olabilir."))
    if anomaly_status == "Anomaly":
        recommendations.append(("error",   "**Isolation Forest Uyarısı:** Bu bölgenin geçmiş 5 yıllık tarihinde sıra dışı suç patlamaları (anomali) yaşanmıştır. İstikrarsız bir bölgedir."))

    if not recommendations:
        st.success("✅ Bu bölge için kritik bir uyarı bulunmamaktadır.")
    else:
        for rtype, rtext in recommendations:
            if rtype == "error":
                st.error(rtext)
            else:
                st.warning(rtext)

    # ── AI Prediction (LOGIC UNCHANGED) ──
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    sec("🔮", "Yapay Zeka ile Gelecek Tahmini")
    st.markdown("<p style='color:#3A4A60; font-size:0.82rem; margin-bottom:14px;'>Seçilen bölge için LSTM modelimizin bir sonraki ay öngörüsü:</p>", unsafe_allow_html=True)

    from src.prediction_helper import predict_next_month
    if st.button("⚡  Bir Sonraki Ayı Tahmin Et"):
        try:
            next_month_score = predict_next_month(selected_area)
            p1, p2 = st.columns([1, 2])
            with p1:
                st.metric("Tahmini Güvenlik Skoru", f"{next_month_score:.2f}")
            with p2:
                if next_month_score > selected_data["safety_score"]:
                    st.success("📈 Tahmin: Güvenlik skorunun artması bekleniyor (İyileşme).")
                else:
                    st.error("📉 Tahmin: Güvenlik skorunun düşmesi bekleniyor (Risk artışı!).")
        except Exception as e:
            st.error(f"Tahmin yapılamadı (Yeterli veri olmayabilir): {e}")

    # ── Top 10 Safe/Risky ──
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    sec("🏆", "En Güvenli & En Riskli Bölgeler")

    top_safe  = df.sort_values("safety_score", ascending=False).head(10)
    top_risky = df.sort_values("safety_score", ascending=True).head(10)

    sc, rc = st.columns(2)
    with sc:
        st.markdown("<p style='color:#10B981; font-family:Barlow Condensed,sans-serif; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>🟢  En Güvenli 10</p>", unsafe_allow_html=True)
        st.dataframe(
            top_safe[["community_name","safety_score","total_crime_count"]]
              .rename(columns={"community_name":"Bölge","safety_score":"Skor","total_crime_count":"Aylık Suç"}),
            use_container_width=True, hide_index=True
        )
    with rc:
        st.markdown("<p style='color:#F43F5E; font-family:Barlow Condensed,sans-serif; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>🔴  En Riskli 10</p>", unsafe_allow_html=True)
        st.dataframe(
            top_risky[["community_name","safety_score","total_crime_count"]]
              .rename(columns={"community_name":"Bölge","safety_score":"Skor","total_crime_count":"Aylık Suç"}),
            use_container_width=True, hide_index=True
        )

    # ── Area Comparison (LOGIC UNCHANGED) ──
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    sec("📊", "Bölge Karşılaştırma Analizi")

    cc1, cc2 = st.columns(2)
    with cc1:
        c1 = st.selectbox("1. Bölge:", community_list, key="comp1")
    with cc2:
        c2 = st.selectbox("2. Bölge:", community_list, key="comp2")

    d1 = df[df["community_name"] == c1].iloc[0]
    d2 = df[df["community_name"] == c2].iloc[0]

    # Custom styled comparison table
    st.markdown(f"""
    <table class="cmp-table">
        <thead>
            <tr>
                <th class="cmp-th">Metrik</th>
                <th class="cmp-th">{c1}</th>
                <th class="cmp-th">{c2}</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="cmp-td cmp-td-metric">Güvenlik Skoru</td>
                <td class="cmp-td"><span style="font-family:'Space Mono',monospace;">{d1['safety_score']:.2f}</span></td>
                <td class="cmp-td"><span style="font-family:'Space Mono',monospace;">{d2['safety_score']:.2f}</span></td>
            </tr>
            <tr>
                <td class="cmp-td cmp-td-metric">Risk Seviyesi</td>
                <td class="cmp-td">{badge_html(d1['risk_level'])}</td>
                <td class="cmp-td">{badge_html(d2['risk_level'])}</td>
            </tr>
            <tr>
                <td class="cmp-td cmp-td-metric">Toplam Suç Sayısı</td>
                <td class="cmp-td"><span style="font-family:'Space Mono',monospace;">{int(d1['total_crime_count']):,}</span></td>
                <td class="cmp-td"><span style="font-family:'Space Mono',monospace;">{int(d2['total_crime_count']):,}</span></td>
            </tr>
            <tr>
                <td class="cmp-td cmp-td-metric">Suç Ağırlığı</td>
                <td class="cmp-td"><span style="font-family:'Space Mono',monospace;">{d1['total_crime_weight']:.2f}</span></td>
                <td class="cmp-td"><span style="font-family:'Space Mono',monospace;">{d2['total_crime_weight']:.2f}</span></td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    # Visual bar comparison
    metrics_keys   = ["safety_score", "total_crime_count", "total_crime_weight"]
    metrics_labels = ["Güvenlik Skoru", "Aylık Suç", "Suç Ağırlığı"]
    chart_data = pd.DataFrame({
        "Metrik": metrics_labels * 2,
        "Değer":  [d1[m] for m in metrics_keys] + [d2[m] for m in metrics_keys],
        "Bölge":  [c1] * 3 + [c2] * 3,
    })
    comp_bar = px.bar(chart_data, x="Metrik", y="Değer", color="Bölge",
                      barmode="group", color_discrete_sequence=["#00D4FF", "#A78BFA"])
    comp_bar = style_chart(comp_bar, f"{c1}  vs  {c2} — Metrik Karşılaştırması")
    st.plotly_chart(comp_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HARİTA
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    sec("🗺️", "Chicago İnteraktif Güvenlik Haritası")
    st.markdown("<p style='color:#3A4A60; font-size:0.82rem; margin-bottom:16px;'>Bölge noktalarına tıklayarak detaylı güvenlik skorlarını görebilirsiniz.</p>", unsafe_allow_html=True)

    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="CartoDB dark_matter")

    for _, row in filtered_df.iterrows():
        color = get_color(row["risk_level"])
        hex_c = RISK_HEX.get(row["risk_level"], "#888")
        popup_html = f"""
        <div style="font-family:'DM Sans',Arial,sans-serif; width:195px; padding:4px;">
            <div style="font-size:0.9rem; font-weight:700; color:#E8EDF5;
                        border-bottom:2px solid {hex_c}; padding-bottom:6px; margin-bottom:8px;">
                {row['community_name']}
            </div>
            <div style="font-size:0.78rem; line-height:1.8; color:#94A3B8;">
                <b style='color:#7A8BA0;'>Güvenlik Skoru</b><br>
                <span style='font-family:monospace; font-size:1.1rem; color:{hex_c};'>{row['safety_score']:.1f}</span><br>
                <b style='color:#7A8BA0;'>Risk Seviyesi</b><br>
                <span style='color:{hex_c}; font-weight:700;'>{row['risk_level']}</span><br>
                <b style='color:#7A8BA0;'>Aylık Suç (Ort.)</b><br>
                {int(row['total_crime_count']):,}
            </div>
        </div>"""
        folium.CircleMarker(
            location=[row["avg_latitude"], row["avg_longitude"]],
            radius=9, color=color, fill=True,
            fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['community_name']} — {row['risk_level']}"
        ).add_to(m)

    heat_data = [[r["avg_latitude"], r["avg_longitude"], r["total_crime_weight"]]
                 for _, r in filtered_df.iterrows()]
    if heat_data:
        HeatMap(heat_data, radius=35, blur=20, max_zoom=1).add_to(m)

    st_html(m._repr_html_(), height=600)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    sec("🤖", "Algoritma Performans Karşılaştırması")
    st.markdown("<p style='color:#3A4A60; font-size:0.82rem; margin-bottom:16px;'>Geleneksel ML algoritmaları ile LSTM (Derin Öğrenme) modelinin 2024 R² başarı oranı kıyaslaması.</p>", unsafe_allow_html=True)

    bar_model = px.bar(
        model_df, x="Model", y="R2 Score", color="Model",
        text=model_df["R2 Score"].apply(lambda x: f"{x:.4f}"),
        color_discrete_sequence=CHART_PALETTE,
    )
    bar_model.update_traces(textposition="outside", textfont=dict(color="#C8D3E0", size=11))
    bar_model = style_chart(bar_model, "Yapay Zeka Modelleri Başarı Sıralaması (R² Score)")
    st.plotly_chart(bar_model, use_container_width=True)

    with st.expander("📋  Tüm Model Metriklerini Tablo Olarak Gör"):
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    sec("🎯", "Gerçek vs. Tahmin Edilen Safety Score")

    selected_model_pred = st.selectbox(
        "Model Seçiniz:", predictions_df["Model"].unique(), key="model_pred_select"
    )
    sel_preds = predictions_df[predictions_df["Model"] == selected_model_pred]

    fig_scatter = px.scatter(
        sel_preds,
        x="Actual Safety Score", y="Predicted Safety Score",
        labels={"Actual Safety Score": "Gerçek Safety Score",
                "Predicted Safety Score": "Tahmin Edilen Safety Score"},
        color_discrete_sequence=["#00D4FF"],
        opacity=0.65,
    )
    fig_scatter.add_shape(
        type="line", line=dict(dash="dash", color="#F43F5E", width=1.5),
        x0=sel_preds["Actual Safety Score"].min(), y0=sel_preds["Actual Safety Score"].min(),
        x1=sel_preds["Actual Safety Score"].max(), y1=sel_preds["Actual Safety Score"].max(),
    )
    fig_scatter = style_chart(fig_scatter, f"{selected_model_pred} — Gerçek vs Tahmin Safety Score")
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("📋  Tahmin Verilerini Detaylı Gör"):
        st.dataframe(sel_preds, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — KÜMELEME
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    cluster_metrics_df    = pd.read_csv(CLUSTER_METRICS_PATH)
    hierarchical_metrics_df = pd.read_csv(HIERARCHICAL_METRICS_PATH)
    k_sil = cluster_metrics_df.loc[cluster_metrics_df["Metric"] == "Silhouette Score", "Value"].values[0]
    h_sil = hierarchical_metrics_df.loc[hierarchical_metrics_df["Metric"] == "Silhouette Score", "Value"].values[0]

    km_tab, hier_tab = st.tabs(["🔵  K-Means Kümeleme", "🌳  Hiyerarşik Kümeleme"])

    with km_tab:
        sec("🔵", "K-Means Kümeleme Analizi")
        st.metric("Silhouette Skoru", f"{k_sil:.4f}")
        cht_km = px.scatter(
            filtered_df, x="total_crime_weight", y="safety_score",
            color="risk_level", hover_data=["community_name"],
            color_discrete_map=RISK_HEX,
            labels={"total_crime_weight": "Suç Ağırlığı", "safety_score": "Güvenlik Skoru",
                    "risk_level": "Risk Seviyesi"},
        )
        cht_km = style_chart(cht_km, "K-Means Dağılım Grafiği — Risk Grupları")
        st.plotly_chart(cht_km, use_container_width=True)

    with hier_tab:
        sec("🌳", "Hiyerarşik Kümeleme Analizi")
        st.metric("Silhouette Skoru", f"{h_sil:.4f}")

        hier_df2 = pd.read_csv(HIERARCHICAL_DATA_PATH)
        hier_df2["community_area"] = hier_df2["community_area"].astype(int)
        hier_df2["community_name"]  = hier_df2["community_area"].map(community_names)
        hier_filtered = hier_df2[
            (hier_df2["hierarchical_risk_level"].isin(risk_filter)) &
            (hier_df2["safety_score"] >= min_score)
        ]

        st.markdown("<p style='color:#3A4A60; font-size:0.78rem; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;'>Dendrogram</p>", unsafe_allow_html=True)
        try:
            dendrogram_image = Image.open(DENDROGRAM_PATH)
            st.image(dendrogram_image, use_container_width=True)
        except:
            st.warning("Dendrogram görseli yüklenemedi.")

        cht_hier = px.scatter(
            hier_filtered, x="total_crime_weight", y="safety_score",
            color="hierarchical_risk_level", hover_data=["community_name"],
            color_discrete_map=RISK_HEX,
            labels={"total_crime_weight": "Suç Ağırlığı", "safety_score": "Güvenlik Skoru",
                    "hierarchical_risk_level": "Risk Seviyesi"},
        )
        cht_hier = style_chart(cht_hier, "Hiyerarşik Kümeleme Dağılım Grafiği")
        st.plotly_chart(cht_hier, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GELİŞMİŞ ANALİZ
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    sec("⚠️", "Zaman Serisi Anomali Tespiti")
    st.markdown("<p style='color:#3A4A60; font-size:0.82rem; margin-bottom:14px;'>5 yıllık dönemde suç oranlarında açıklanamayan ani patlamaların yaşandığı bölge ve dönemler (Isolation Forest).</p>", unsafe_allow_html=True)

    anomalies_only    = anomaly_df[anomaly_df["anomaly_status"] == "Anomaly"]
    anomalies_display = anomalies_only[["community_name","year_month","safety_score",
                                        "total_crime_count","total_crime_weight"]]
    st.dataframe(
        anomalies_display.sort_values("total_crime_weight", ascending=False).head(20)
        .rename(columns={"community_name":"Bölge","year_month":"Dönem",
                         "safety_score":"Güvenlik Skoru","total_crime_count":"Suç Sayısı",
                         "total_crime_weight":"Suç Ağırlığı"}),
        use_container_width=True, hide_index=True
    )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    sec("🧠", "Özellik Önemi — Feature Importance")
    st.markdown("<p style='color:#3A4A60; font-size:0.82rem; margin-bottom:14px;'>Modeller karar verirken en çok hangi özelliklere dikkat etti?</p>", unsafe_allow_html=True)

    selected_imp_model = st.selectbox("Algoritma Seçiniz:", feature_importance_df["Model"].unique())
    sel_imp_df = feature_importance_df[feature_importance_df["Model"] == selected_imp_model]

    cht_imp = px.bar(
        sel_imp_df, x="Importance", y="Feature", orientation="h",
        color="Feature",
        text=sel_imp_df["Importance"].apply(lambda x: f"{x:.4f}"),
        color_discrete_sequence=CHART_PALETTE,
    )
    cht_imp.update_traces(textposition="outside", textfont=dict(color="#C8D3E0", size=11))
    cht_imp = style_chart(cht_imp, f"{selected_imp_model} — Karar Mekanizması")
    st.plotly_chart(cht_imp, use_container_width=True)