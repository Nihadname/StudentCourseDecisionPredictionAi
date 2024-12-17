from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load('course_predictor_model.pkl')
le = joblib.load('label_encoder.pkl')

# Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Define the feature columns
numeric_features = ['age', 'isParent', 'childAge']

@app.route('/predict', methods=['POST'])
def predict_course():
    try:
        # Parse JSON input
        input_data = request.get_json()

        # Check for required fields
        required_fields = numeric_features
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

        # Create a DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Make the prediction
        prediction = model.predict(input_df)
        predicted_course = le.inverse_transform(prediction)[0]  # Get the course name from the label encoder

        # Return the predicted course as JSON
        return jsonify({"predicted_course": predicted_course})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")  # Host set to 0.0.0.0 for all network interfaces
