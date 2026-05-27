import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


INPUT_PATH = "data/processed/community_clusters.csv"


def load_data():
    return pd.read_csv(INPUT_PATH)


def prepare_features(df):

    X = df[[
        "total_crime_count",
        "total_crime_weight",
        "avg_latitude",
        "avg_longitude"
    ]]

    y = df["safety_score"]

    return X, y


def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print("\nModel Performansı")
    print("-------------------")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return predictions


def visualize_results(y_test, predictions):

    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, predictions)

    plt.xlabel("Gerçek Safety Score")
    plt.ylabel("Tahmin Edilen Safety Score")

    plt.title("Random Forest Tahmin Sonuçları")

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    plt.show()


if __name__ == "__main__":

    df = load_data()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = train_model(X_train, y_train)

    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    visualize_results(y_test, predictions)