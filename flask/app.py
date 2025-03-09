import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# ✅ Load Model (Ensures It Exists)
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("⚠️ model.pkl not found! Retrain the model first.")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ✅ Load Dataset & Scale Features
DATASET_PATH = "diabetes.csv"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("⚠️ diabetes.csv not found! Upload the dataset.")

dataset = pd.read_csv(DATASET_PATH)
dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values

sc = MinMaxScaler(feature_range=(0, 1))
sc.fit(dataset_X)  # Fit the scaler only once

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles form input and returns predictions"""
    try:
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        transformed_features = sc.transform(final_features)
        prediction = model.predict(transformed_features)

        output = "You have Diabetes, please consult a Doctor." if prediction == 1 else "You don't have Diabetes."
        return render_template("index.html", prediction_text=output)

    except Exception as e:
        return render_template("index.html", prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
