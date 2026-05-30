import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hedef dosyamızı güncel panel veri olarak ayarladık
INPUT_PATH = "data/processed/community_safety_scores.csv"
OUTPUT_PATH = "outputs/model_comparison_results.csv"
PREDICTIONS_PATH = "outputs/model_predictions.csv"

def load_data():
    print(f"Model karşılaştırması için panel veri yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def prepare_features(df):
    # Veriyi zaman serisi mantığına uygun olarak kronolojik sıralıyoruz
    df = df.sort_values(by=["community_area", "year_month"]).reset_index(drop=True)

    # Modeller için Geçmiş Ay Hafızası (Lag Features)
    df["lag_1"] = df.groupby("community_area")["safety_score"].shift(1)
    df["lag_2"] = df.groupby("community_area")["safety_score"].shift(2)
    df["lag_3"] = df.groupby("community_area")["safety_score"].shift(3)

    # İlk 3 ayın geçmişi olmadığı için (NaN) o satırları siliyoruz
    df_model = df.dropna(subset=["lag_1", "lag_2", "lag_3"]).copy()

    # Veri sızıntısını önlemek için crime_weight veya count sütunları ÇIKARILDI
    X = df_model[[
        "avg_latitude",
        "avg_longitude",
        "month",
        "year",
        "lag_1",
        "lag_2",
        "lag_3"
    ]]
    
    y = df_model["safety_score"]
    
    # Train-Test ayırımı için yılları ayrı bir dizi olarak tutuyoruz
    target_years = df_model["year"].values
    
    return X, y, target_years

if __name__ == "__main__":
    df = load_data()
    X, y, target_years = prepare_features(df)

    # Geleceği test etmek için 2024 yılını tamamen saklıyoruz
    train_mask = target_years < 2024
    test_mask = target_years >= 2024

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"Eğitim Verisi: {len(X_train)} satır | Test Verisi (2024): {len(X_test)} satır")
    print("Tüm modeller arenaya çıkıyor, eğitilip karşılaştırılıyor...\n")

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()), 
            ("model", LinearRegression())
        ]),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR())
        ]),
        "ANN / MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=2000,
                random_state=42
            ))
        ])
    }

    results = []
    all_predictions = []

    for model_name, model in models.items():
        # Modeli eğit
        model.fit(X_train, y_train)
        
        # 2024 yılı için tahmin yap
        predictions = model.predict(X_test)

        # Hata metriklerini hesapla
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": model_name,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2 Score": round(r2, 4)
        })

        for actual, predicted in zip(y_test.values, predictions):
            all_predictions.append({
                "Model": model_name,
                "Actual Safety Score": round(actual, 4),
                "Predicted Safety Score": round(predicted, 4)
            })

    # Sonuçları DataFrame'e çevir ve R2 Skoruna göre büyükten küçüğe sırala
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="R2 Score", ascending=False)
    
    predictions_df = pd.DataFrame(all_predictions)

    print("="*65)
    print("                 MODEL PERFORMANS KARŞILAŞTIRMASI")
    print("="*65)
    print(results_df.to_string(index=False))
    print("="*65)

    Path("outputs").mkdir(exist_ok=True)

    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[Başarılı] Karşılaştırma tablosu kaydedildi: {OUTPUT_PATH}")
    print(f"[Başarılı] Tahmin geçmişi kaydedildi: {PREDICTIONS_PATH}")