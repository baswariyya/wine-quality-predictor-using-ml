import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

def train_wine_model(csv_path, model_name, scaler_name, epochs=50, random_state=42):
    print(f"\n=== Training model for {csv_path} ===")
    
    data = pd.read_csv(csv_path, sep=';')
    X = data.drop(columns=["quality"])
    y = data["quality"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, verbose=1)

    model.save(model_name)
    joblib.dump(scaler, scaler_name)

    print(f"{model_name} and {scaler_name} saved!")

# Training both models
train_wine_model("winequality-red.csv", "model_red.keras", "scaler_red.save")
train_wine_model("winequality-white.csv", "model_white.keras", "scaler_white.save")

# Saving artifact dictionary
artifacts = {
    "red": {"model_path": "model_red.keras", "scaler_path": "scaler_red.save"},
    "white": {"model_path": "model_white.keras", "scaler_path": "scaler_white.save"}
}
joblib.dump(artifacts, "wine_models_paths.save")
print("All artifacts saved to 'wine_models_paths.save'")
