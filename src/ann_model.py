import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

INPUT_PATH = "data/processed/community_safety_scores.csv"

def load_data():
    print(f"ANN için veri yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def prepare_features(df):
    # Veriyi kronolojik sıralıyoruz
    df = df.sort_values(by=["community_area", "year_month"]).reset_index(drop=True)

    # Random Forest'ta yaptığımız Lag Features (Hafıza) mantığı
    df["lag_1"] = df.groupby("community_area")["safety_score"].shift(1)
    df["lag_2"] = df.groupby("community_area")["safety_score"].shift(2)
    df["lag_3"] = df.groupby("community_area")["safety_score"].shift(3)

    # NaN olan ilk ayları uçuruyoruz
    df_model = df.dropna(subset=["lag_1", "lag_2", "lag_3"]).copy()

    # Özellikler (X) ve Hedef (y)
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
    
    # 2024 yılını (Test setini) ayırmak için yılları da döndürüyoruz
    target_years = df_model["year"].values
    
    return X, y, target_years

def scale_data(X_train, X_test, y_train, y_test):
    # DİKKAT: Veri sızıntısını önlemek için Scaler SADECE Eğitim verisiyle (fit) eğitilir!
    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler

def build_and_train_ann(X_train, y_train):
    print("ANN (Yapay Sinir Ağı) inşa ediliyor ve eğitiliyor...")
    
    model = Sequential()
    
    # Girdi Katmanı ve İlk Gizli Katman (Hidden Layer)
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.2))
    
    # İkinci Gizli Katman
    model.add(Dense(32, activation='relu'))
    
    # Çıktı Katmanı (Tek tahmin)
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=100, # Basit ağlar daha hızlı eğitildiği için epoch'u yüksek tutabiliriz
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model

def evaluate_and_visualize(model, X_test, y_test_scaled, y_scaler):
    # Ölçeklenmiş tahminleri al
    scaled_predictions = model.predict(X_test)
    
    # Tahminleri ve gerçek değerleri orijinal 0-100 aralığına geri çevir
    predictions = y_scaler.inverse_transform(scaled_predictions)
    y_test_real = y_scaler.inverse_transform(y_test_scaled)
    
    mae = mean_absolute_error(y_test_real, predictions)
    mse = mean_squared_error(y_test_real, predictions)
    r2 = r2_score(y_test_real, predictions)
    
    print("\n" + "="*35)
    print("      ANN MODEL PERFORMANSI")
    print("="*35)
    print(f"Test Veri Seti (2024 Yılı): {len(y_test_real)} tahmin")
    print(f"MAE (Ortalama Mutlak Hata): {mae:.4f}")
    print(f"MSE (Ortalama Kare Hata): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("="*35 + "\n")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_real, predictions, alpha=0.5, color='orange')
    plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
    plt.xlabel("Gerçek Safety Score (2024)")
    plt.ylabel("ANN Tahmini")
    plt.title("ANN (Multi-Layer Perceptron): Gerçek vs Tahmin")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    df = load_data()
    
    X, y, target_years = prepare_features(df)
    
    # 2024 öncesi Train, 2024 Test (Zaman sızıntısı yok)
    train_mask = target_years < 2024
    test_mask = target_years >= 2024
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Eğitim verisi: {len(X_train)} satır | Test verisi (2024): {len(X_test)} satır")
    
    # Sinir ağları için ölçeklendirme
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_data(X_train, X_test, y_train, y_test)
    
    # Model Eğitimi
    model = build_and_train_ann(X_train_scaled, y_train_scaled)
    
    # Değerlendirme
    evaluate_and_visualize(model, X_test_scaled, y_test_scaled, y_scaler)