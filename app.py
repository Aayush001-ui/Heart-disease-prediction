from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)


# Load Model & Scaler
MODEL_PATH = "heart_disease_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or Scaler file not found. Train first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature order
expected_features = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]


# Routes
@app.route('/')
def home():
    return render_template("index.html", show_prediction_section=False)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = []

        # ✅ Input validation (important)
        for feat in expected_features:
            val = request.form.get(feat)

            if val is None or val == "":
                raise ValueError(f"Missing value for {feat}")

            values.append(float(val))

        features = np.array([values])

        # Always scale (clean pipeline)
        features_scaled = scaler.transform(features)

        #  Prediction + confidence
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        confidence = round(probability * 100, 2)

        #  Result
        if prediction == 1:
            result_text = f"⚠️ High Risk of Heart Disease ({confidence}% confidence)"
        else:
            result_text = f"✅ Low Risk of Heart Disease ({confidence}% confidence)"

        return render_template(
            "index.html",
            prediction_text=result_text,
            confidence=confidence,
            show_prediction_section=True
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            show_prediction_section=True
        )


if __name__ == '__main__':
    app.run(debug=True)