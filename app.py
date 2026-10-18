import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Resolve absolute paths to model file
MODEL_PATH = os.path.join(BASE_DIR, 'iris_model.pkl')

# Load the trained model once when the app starts
model = joblib.load(MODEL_PATH)

# Iris species class names
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

# Required model feature names
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

# Field alias mapping for user convenience (supports both short and full names)
FIELD_ALIASES = {
    "sepal_length": "sepal length (cm)",
    "sepal length": "sepal length (cm)",
    "sepal length (cm)": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "sepal width": "sepal width (cm)",
    "sepal width (cm)": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal length": "petal length (cm)",
    "petal length (cm)": "petal length (cm)",
    "petal_width": "petal width (cm)",
    "petal width": "petal width (cm)",
    "petal width (cm)": "petal width (cm)",
}


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # If GET, return documentation on how to use the endpoint
    if request.method == 'GET':
        # Also support query parameters on GET for simple test queries
        args = request.args
        if args and any(k in args for k in ["sepal length (cm)", "sepal_length"]):
            data = dict(args)
        else:
            return jsonify({
                "message": "Iris Taxonomic Classification API. Send a POST request with JSON payload or query parameters.",
                "model": "Gaussian Naive Bayes",
                "reference": "R.A. Fisher (1936)",
                "required_features": FEATURE_NAMES,
                "classes": CLASS_NAMES,
                "example_request": {
                    "sepal length (cm)": 5.1,
                    "sepal width (cm)": 3.5,
                    "petal length (cm)": 1.4,
                    "petal width (cm)": 0.2
                }
            }), 200
    else:
        # Check if request has JSON body
        data = request.get_json(silent=True)
        if data is None:
            # Fallback to form data if JSON wasn't provided
            if request.form:
                data = dict(request.form)
            else:
                return jsonify({
                    "error": "Invalid request. Please provide JSON with Content-Type: application/json."
                }), 400

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    # Normalize fields using aliases
    normalized_data = {}
    for key, val in data.items():
        clean_key = str(key).strip().lower()
        if clean_key in FIELD_ALIASES:
            normalized_data[FIELD_ALIASES[clean_key]] = val
        else:
            normalized_data[key] = val

    # Validate all required fields are present
    missing = [f for f in FEATURE_NAMES if f not in normalized_data]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing)}",
            "required_fields": FEATURE_NAMES
        }), 400

    # Validate numeric types and ranges
    feature_values = []
    for f in FEATURE_NAMES:
        raw_val = normalized_data[f]
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            return jsonify({
                "error": f"Field '{f}' must be a valid number, got: {repr(raw_val)}"
            }), 400

        if val <= 0:
            return jsonify({
                "error": f"Field '{f}' must be greater than 0, got: {val}"
            }), 400

        if val > 100:
            return jsonify({
                "error": f"Field '{f}' exceeds realistic bounds (max 100 cm), got: {val}"
            }), 400

        feature_values.append(val)

    try:
        # Convert to DataFrame with proper column names
        custom_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

        # Make prediction
        prediction = model.predict(custom_df)[0]
        prediction_int = int(prediction)

        # Safe class indexing
        if 0 <= prediction_int < len(CLASS_NAMES):
            class_flower = CLASS_NAMES[prediction_int]
        else:
            class_flower = f"Class {prediction_int}"

        # Probabilities
        response = {
            "prediction": prediction_int,
            "class_flower": class_flower
        }

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(custom_df)[0]
            response["probabilities"] = {
                CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)