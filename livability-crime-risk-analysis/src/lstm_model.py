import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


INPUT_PATH = "data/processed/community_safety_scores.csv"


def create_sequences(data, lookback=3):
    X, y = [], []

    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])

    return np.array(X), np.array(y)


df = pd.read_csv(INPUT_PATH)

monthly_df = df.groupby(["year", "month"], as_index=False)["safety_score"].mean()
monthly_df = monthly_df.sort_values(["year", "month"])

values = monthly_df["safety_score"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

lookback = 3
X, y = create_sequences(scaled_values, lookback)

X = X.reshape((X.shape[0], X.shape[1], 1))

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

model = Sequential()
model.add(LSTM(32, activation="relu", input_shape=(lookback, 1)))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    verbose=1
)

predictions = model.predict(X_test)

y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_real = scaler.inverse_transform(predictions)

mae = mean_absolute_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)

print("\nLSTM Model Performansı")
print("----------------------")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")




plt.figure(figsize=(8, 6))
plt.plot(y_test_real, label="Gerçek Safety Score")
plt.plot(predictions_real, label="LSTM Tahmini")
plt.xlabel("Zaman")
plt.ylabel("Safety Score")
plt.title("LSTM Safety Score Tahmin Sonuçları")
plt.legend()
plt.show()