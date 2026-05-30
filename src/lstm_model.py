import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TensorFlow/Keras Kütüphaneleri
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

INPUT_PATH = "data/processed/community_safety_scores.csv"

def load_data():
    print(f"LSTM için veri yükleniyor: {INPUT_PATH}")
    return pd.read_csv(INPUT_PATH)

def prepare_lstm_data(df, lookback=3):
    print(f"Veri LSTM formatına (3 Boyutlu Tensör) dönüştürülüyor... Lookback: {lookback} ay")
    
    # 1. Veriyi sırala
    df = df.sort_values(by=["community_area", "year_month"]).reset_index(drop=True)
    
    # 2. Sinir Ağları için Veriyi Ölçeklendir (0 ile 1 arasına sıkıştır)
    scaler = MinMaxScaler()
    df["scaled_score"] = scaler.fit_transform(df[["safety_score"]])
    
    X, y, target_years = [], [], []
    
    # 3. Her mahalle için kayan pencere (sliding window) oluştur
    for area in df["community_area"].unique():
        area_df = df[df["community_area"] == area]
        scores = area_df["scaled_score"].values
        years = area_df["year"].values
        
        # Lookback kadar geriye bakıp bir sonrakini tahmin et
        for i in range(len(scores) - lookback):
            X.append(scores[i : i + lookback])       # Girdi: Örn. Ocak, Şubat, Mart
            y.append(scores[i + lookback])           # Çıktı: Örn. Nisan
            target_years.append(years[i + lookback]) # Ayırmak için tahmin edilen yılı tut
            
    # Listeleri Numpy Dizilerine çevir ve X'i 3 boyutlu yap: [Örnek, Zaman Adımı, Özellik=1]
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    target_years = np.array(target_years)
    
    return X, y, target_years, scaler

def build_and_train_lstm(X_train, y_train):
    print("LSTM Sinir Ağı inşa ediliyor ve eğitiliyor...")
    
    model = Sequential()
    
    # Girdi Katmanı ve LSTM Hücreleri
    model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    
    # Aşırı Öğrenmeyi (Overfitting) Engellemek için Dropout
    model.add(Dropout(0.2))
    
    # Çıktı Katmanı (Tek bir tahmin değeri)
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    # Early Stopping: Model gelişmeyi durdurursa eğitimi erken keser
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Modeli Eğit (Validation split ile kendi içinde de %20'lik bir doğrulama yapar)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1 # Eğitim sürecini terminalde gösterir
    )
    
    return model, history

def evaluate_and_visualize(model, X_test, y_test, scaler):
    # Tahmin yap (0 ile 1 arasında değerler dönecek)
    scaled_predictions = model.predict(X_test)
    
    # Sıkıştırılmış değerleri orijinal 0-100 güvenlik skoru formatına geri döndür (Inverse Transform)
    predictions = scaler.inverse_transform(scaled_predictions)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test_real, predictions)
    mse = mean_squared_error(y_test_real, predictions)
    r2 = r2_score(y_test_real, predictions)
    
    print("\n" + "="*35)
    print("      LSTM MODEL PERFORMANSI")
    print("="*35)
    print(f"Test Veri Seti (2024 Yılı): {len(y_test)} tahmin")
    print(f"MAE (Ortalama Mutlak Hata): {mae:.4f}")
    print(f"MSE (Ortalama Kare Hata): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("="*35 + "\n")
    
    # Görselleştirme
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_real, predictions, alpha=0.6, color='purple')
    plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
    plt.xlabel("Gerçek Safety Score (2024)")
    plt.ylabel("LSTM Tahmini")
    plt.title("LSTM Derin Öğrenme: Gerçek vs Tahmin")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Mevcut tahminleri CSV olarak kaydedelim (app.py için)
    lstm_results = pd.DataFrame({
        "Model": ["LSTM (Deep Learning)"] * len(y_test_real),
        "Actual Safety Score": y_test_real.flatten(),
        "Predicted Safety Score": predictions.flatten()
    })
    lstm_results.to_csv("outputs/lstm_predictions.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    df = load_data()
    
    # Lookback=3 diyerek geçmiş 3 aya bakıyoruz. İstersen 6 yaparak hafızayı uzatabilirsin.
    X, y, target_years, scaler = prepare_lstm_data(df, lookback=3)
    
    # Zaman yolculuğunu önle: 2024 öncesi Train, 2024 ve sonrası Test
    train_mask = target_years < 2024
    test_mask = target_years >= 2024
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Eğitim verisi: {len(X_train)} pencere | Test verisi (2024): {len(X_test)} pencere")
    
    # Modeli Eğit
    model, history = build_and_train_lstm(X_train, y_train)
    # Modelin içine ekle:
    import pickle
    # Modeli kaydet
    model.save("outputs/lstm_model.keras")
    
    # Ölçekleyiciyi (Scaler) kaydet (Çok Önemli! Veriyi nasıl küçülttüğünü bilmezsek tahmini geri çeviremeyiz)
    with open("outputs/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print("\nModel ve Scaler başarıyla kaydedildi!")
    # Test Et ve Görselleştir
    evaluate_and_visualize(model, X_test, y_test, scaler)