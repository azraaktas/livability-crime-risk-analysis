import pandas as pd
import folium
from pathlib import Path

# K-Means aşamasından çıkan 77 mahallelik risk profilini okuyoruz
INPUT_PATH = "data/processed/community_clusters.csv"
OUTPUT_PATH = "outputs/chicago_safety_map.html"

def load_data():
    print(f"Harita için kümeleme verisi yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def get_risk_color(risk_level):
    """Risk seviyesine göre harita marker rengini belirler."""
    if risk_level == "Safe":
        return "green"
    elif risk_level == "Medium Risk":
        return "orange"
    else:
        return "red"

def create_safety_map(df):
    print("Chicago Güvenlik ve Risk Haritası oluşturuluyor...")
    
    # Isı haritasında olduğu gibi koyu tema (dark_matter) kullanarak marker renklerini öne çıkarıyoruz
    chicago_map = folium.Map(
        location=[41.8781, -87.6298],
        zoom_start=11,
        tiles="CartoDB dark_matter"
    )

    for _, row in df.iterrows():
        color = get_risk_color(row["risk_level"])
        
        # Profesyonel bir görünüm için Popup içeriğini şık bir HTML tablosuna dönüştürüyoruz
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 200px; color: #333;">
            <h4 style="margin: 0 0 10px 0; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                Bölge: ID {int(row['community_area'])}
            </h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 3px 0; font-weight: bold;">Güvenlik Skoru:</td>
                    <td style="text-align: right;">{row['safety_score']:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 3px 0; font-weight: bold;">Risk Seviyesi:</td>
                    <td style="text-align: right; color: {color}; font-weight: bold;">{row['risk_level']}</td>
                </tr>
                <tr>
                    <td style="padding: 3px 0; font-weight: bold;">Ort. Aylık Suç:</td>
                    <td style="text-align: right;">{int(row['total_crime_count'])}</td>
                </tr>
            </table>
        </div>
        """
        
        # Kullanıcı fareyi marker üzerine getirdiğinde göreceği pratik bilgi (Tooltip)
        tooltip_text = f"Bölge {int(row['community_area'])} - {row['risk_level']}"

        folium.CircleMarker(
            location=[row["avg_latitude"], row["avg_longitude"]],
            radius=9, # Görünürlüğü artırmak için hafifçe büyütüldü
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=tooltip_text
        ).add_to(chicago_map)

    return chicago_map

if __name__ == "__main__":
    df = load_data()
    
    safety_map = create_safety_map(df)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    safety_map.save(OUTPUT_PATH)

    print(f"\nİnteraktif risk haritası başarıyla kaydedildi: {OUTPUT_PATH}")
    print("Bu HTML dosyasını tarayıcınızda açarak mahallelerin üzerine tıklayabilirsiniz.")