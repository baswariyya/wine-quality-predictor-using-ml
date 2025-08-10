import joblib
import tensorflow as tf

# Loading artifact paths
artifacts = joblib.load("wine_models_paths.save")

# Getting wine type
wine_type = input("Enter wine type (red/white): ").strip().lower()
if wine_type not in artifacts:
    print("Invalid wine type. Please enter 'red' or 'white'.")
    exit()

# Loading model and scaler
model = tf.keras.models.load_model(artifacts[wine_type]["model_path"])
scaler = joblib.load(artifacts[wine_type]["scaler_path"])

# Feature listing
features = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

# Getting user input for each feature
values = []
for f in features:
    while True:
        try:
            val = float(input(f"Enter {f}: ").strip())
            values.append(val)
            break
        except ValueError:
            print(" Please enter a valid number.")

# Scale and predicting the quality of wine
values_scaled = scaler.transform([values])
prediction = model.predict(values_scaled, verbose=0)

print(f"\n Predicted quality for {wine_type} wine: {prediction[0][0]:.2f}")
