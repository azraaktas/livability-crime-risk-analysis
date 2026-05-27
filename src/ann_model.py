import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path


INPUT_PATH = "data/processed/community_clusters.csv"
OUTPUT_PATH = "outputs/ann_predictions.csv"


df = pd.read_csv(INPUT_PATH)

X = df[[
    "total_crime_count",
    "total_crime_weight",
    "avg_latitude",
    "avg_longitude"
]]

y = df["safety_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ann", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nANN / MLP Model Performansı")
print("----------------------------")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

results_df = pd.DataFrame({
    "Actual Safety Score": y_test.values,
    "Predicted Safety Score": predictions
})

Path("outputs").mkdir(exist_ok=True)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nANN sonuçları kaydedildi: {OUTPUT_PATH}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.xlabel("Gerçek Safety Score")
plt.ylabel("Tahmin Edilen Safety Score")
plt.title("ANN / MLP Tahmin Sonuçları")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.show()