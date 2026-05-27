import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


INPUT_PATH = "data/processed/community_clusters.csv"
OUTPUT_PATH = "outputs/model_comparison_results.csv"
PREDICTIONS_PATH = "outputs/model_predictions.csv"


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

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ),

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

predictions_df = pd.DataFrame(all_predictions)

print("\nModel Performans Karşılaştırması")
print("--------------------------------")
print(results_df.to_string(index=False))

Path("outputs").mkdir(exist_ok=True)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nModel karşılaştırma sonuçları kaydedildi: {OUTPUT_PATH}")
print(f"Model tahmin sonuçları kaydedildi: {PREDICTIONS_PATH}")