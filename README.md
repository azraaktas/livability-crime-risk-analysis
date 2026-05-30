# Yaşanabilirlik Odaklı Güvenlik Skoru ve Suç Analiz Sistemi

Bu proje, Chicago suç verilerini API üzerinden çekerek mahalle/bölge bazlı güvenlik skoru üretir, bölgeleri risk seviyelerine göre sınıflandırır ve makine öğrenmesi modelleri ile güvenlik skorunu tahmin eder.

## Proje Amacı

Amaç, suç verilerini yalnızca toplam suç sayısına göre değil; suç türü, suç ağırlığı, zaman ve konum bilgisiyle birlikte analiz ederek her bölge için 0-100 arası bir güvenlik skoru üretmektir.

## Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Folium
- Plotly
- Streamlit
- Matplotlib

## Kullanılan Modeller

### Kümeleme

- K-Means Clustering

### Regresyon / Tahmin Modelleri

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- SVR
- ANN / MLP Regressor
- LSTM

## Proje Akışı

1. Chicago suç verilerinin API üzerinden çekilmesi
2. Verinin temizlenmesi
3. Suç türlerine ağırlık verilmesi
4. Mahalle bazlı güvenlik skoru hesaplanması
5. K-Means ile risk sınıflandırması
6. Makine öğrenmesi modelleriyle güvenlik skoru tahmini
7. Harita, heatmap ve grafiklerle görselleştirme
8. Streamlit dashboard ile kullanıcı arayüzü oluşturma

## Klasör Yapısı

```text
livability-crime-risk-analysis/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│
└── src/
    ├── data_collection.py
    ├── data_cleaning.py
    ├── safety_score.py
    ├── clustering.py
    ├── map_visualization.py
    ├── heatmap_visualization.py
    ├── random_forest_model.py
    ├── ann_model.py
    └── model_comparison.py

    ## Çalıştırma Adımları

Önce sanal ortam aktif edilir:

```bash
source venv/bin/activate
```

Gerekli kütüphaneler yüklenir:

```bash
pip install -r requirements.txt
```

Veri çekilir:

```bash
python3 src/data_collection.py
```

Veri temizlenir:

```bash
python3 src/data_cleaning.py
```

Güvenlik skoru hesaplanır:

```bash
python3 src/safety_score.py
```

K-Means kümeleme yapılır:

```bash
python3 src/clustering.py
```

Model karşılaştırması yapılır:

```bash
python3 src/model_comparison.py
```

Dashboard çalıştırılır:

```bash
streamlit run app.py
```

---

## Dashboard İçeriği

Dashboard üzerinde:

- Bölge bazlı güvenlik skoru
- Risk seviyesi
- Suç sayısı
- Risk dağılım grafiği
- Safety score dağılım grafiği
- K-Means cluster grafiği
- Silhouette Score
- Chicago risk haritası
- HeatMap
- Model performans karşılaştırması
- Gerçek ve tahmin edilen safety score karşılaştırması

gösterilmektedir.

---

## Performans Metrikleri

Regresyon modelleri için:

- MAE
- MSE
- RMSE
- R2 Score

Kümeleme modeli için:

- Silhouette Score

kullanılmıştır.

---

## Veri Kaynağı

Chicago Data Portal - Crimes 2001 to Present

https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2

---

