import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

# Veri kaynağımız artık panel veri
INPUT_PATH = "data/processed/community_safety_scores.csv"
OUTPUT_PATH = "outputs/feature_importance.csv"

def load_data():
    print(f"Feature Importance için veri yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def prepare_features(df):
    # Veriyi kronolojik sıralıyoruz
    df = df.sort_values(by=["community_area", "year_month"]).reset_index(drop=True)

    # Modellerimizde kullandığımız Lag Features (Hafıza) ekleniyor
    df["lag_1"] = df.groupby("community_area")["safety_score"].shift(1)
    df["lag_2"] = df.groupby("community_area")["safety_score"].shift(2)
    df["lag_3"] = df.groupby("community_area")["safety_score"].shift(3)

    # NaN satırları sil
    df_model = df.dropna(subset=["lag_1", "lag_2", "lag_3"]).copy()

    # DİKKAT: Veri sızıntısı yapacak crime_weight gibi sütunlar yok. 
    # Sadece konum, zaman ve geçmiş hafıza var.
    features = [
        "avg_latitude",
        "avg_longitude",
        "month",
        "year",
        "lag_1",
        "lag_2",
        "lag_3"
    ]

    X = df_model[features]
    y = df_model["safety_score"]
    
    return X, y, features, df_model["year"].values

if __name__ == "__main__":
    df = load_data()
    X, y, features, target_years = prepare_features(df)

    # Zaman sızıntısını önlemek için yine 2024 yılını ayırıyoruz
    train_mask = target_years < 2024
    X_train = X[train_mask]
    y_train = y[train_mask]

    print("Modeller eğitilip özellik önemleri hesaplanıyor...")

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    importance_results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        importances = model.feature_importances_

        for feature, importance in zip(features, importances):
            importance_results.append({
                "Model": model_name,
                "Feature": feature,
                "Importance": round(float(importance), 4)
            })

    importance_df = pd.DataFrame(importance_results)

    # Her model için kendi içinde büyükten küçüğe sırala
    importance_df = importance_df.sort_values(
        by=["Model", "Importance"],
        ascending=[True, False]
    )

    Path("outputs").mkdir(exist_ok=True)

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "="*40)
    print("      FEATURE IMPORTANCE SONUÇLARI")
    print("="*40)
    print(importance_df.to_string(index=False))

    print(f"\nFeature importance başarıyla kaydedildi: {OUTPUT_PATH}")