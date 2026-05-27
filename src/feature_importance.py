import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor


INPUT_PATH = "data/processed/community_clusters.csv"
OUTPUT_PATH = "outputs/feature_importance.csv"


df = pd.read_csv(INPUT_PATH)

features = [
    "total_crime_count",
    "total_crime_weight",
    "avg_latitude",
    "avg_longitude"
]

X = df[features]
y = df["safety_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ),
     "Decision Tree": DecisionTreeRegressor(
        random_state=42
    )
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

print("\nFeature Importance Sonuçları")
print("-----------------------------")
print(importance_df.to_string(index=False))

print(f"\nFeature importance kaydedildi: {OUTPUT_PATH}")