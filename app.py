from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load the trained model once when the app starts
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return "<p>Welcome to the Iris Classification API!</p>"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Required feature names
        required_fields = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

        # Validate that all required fields are present
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing fields in request data"}), 400
        
        # Convert JSON data to DataFrame (as model expects)
        custom_df = pd.DataFrame([[data["sepal length (cm)"], data["sepal width (cm)"], 
                                   data["petal length (cm)"], data["petal width (cm)"]]], 
                                 columns=required_fields)

        # Make prediction
        prediction = model.predict(custom_df)[0]  # Extract single value from array

        # Convert NumPy integer to Python integer
        prediction_int = int(prediction)

        # Load class names
        iris = load_iris()
        class_flower = iris['target_names'][prediction_int]  # ✅ Convert prediction to int before indexing

        return jsonify({
            "prediction": prediction_int,
            "class_flower": class_flower
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
