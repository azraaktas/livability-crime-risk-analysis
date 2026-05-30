import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

INPUT_PATH = "data/processed/community_safety_scores.csv"

def load_data():
    print(f"Model için veri yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def prepare_features(df):
    # Veriyi zaman serisi mantığına uygun olarak kronolojik sıralıyoruz
    df = df.sort_values(by=["community_area", "year_month"]).reset_index(drop=True)

    # HAYAT KURTARAN DOKUNUŞ: Lag Features (Geçmiş Ayların Hafızası)
    # Her mahalle için geçmiş 1, 2 ve 3 ayın güvenlik skorlarını yeni sütunlar olarak ekliyoruz
    df["lag_1"] = df.groupby("community_area")["safety_score"].shift(1)
    df["lag_2"] = df.groupby("community_area")["safety_score"].shift(2)
    df["lag_3"] = df.groupby("community_area")["safety_score"].shift(3)

    # İlk 3 ayın geçmişi olmadığı için (NaN) o satırları siliyoruz
    df_model = df.dropna(subset=["lag_1", "lag_2", "lag_3"]).copy()

    # Model artık hem konumu, hem zamanı, hem de geçmiş 3 ayın trendini biliyor
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
    
    return X, y, df_model

def train_model(X_train, y_train):
    print("Random Forest (Lag Features ile) eğitiliyor, lütfen bekleyin...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1 
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n" + "="*35)
    print("      YENİ MODEL PERFORMANSI")
    print("="*35)
    print(f"Test Veri Seti Boyutu: {len(y_test)} satır")
    print(f"MAE (Ortalama Mutlak Hata): {mae:.4f}")
    print(f"MSE (Ortalama Kare Hata): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("="*35 + "\n")
    
    return predictions

def visualize_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_test, predictions, alpha=0.5, color='green')
    
    plt.xlabel("Gerçek Safety Score", fontsize=12)
    plt.ylabel("Tahmin Edilen Safety Score", fontsize=12)
    plt.title("Random Forest (Hafızalı Model): Gerçek vs Tahmin", fontsize=14)
    
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--",
        color="red",
        linewidth=2,
        label="Kusursuz Tahmin Çizgisi"
    )
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    df = load_data()
    
    X, y, _ = prepare_features(df)
    
    # shuffle=False kuralımız hala geçerli, geleceği geçmişe karıştırmıyoruz.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=False 
    )
    
    print(f"Eğitim verisi: {X_train.shape[0]} satır | Test verisi: {X_test.shape[0]} satır")
    
    model = train_model(X_train, y_train)
    
    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )
    
    visualize_results(y_test, predictions)