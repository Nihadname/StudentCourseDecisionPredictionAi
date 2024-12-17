from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load('course_predictor_model.pkl')
le = joblib.load('label_encoder.pkl')

# Flask App
app = Flask(__name__)
CORS(app)

# Define the exact feature columns as used during training
feature_columns = ['age', 'studytime', 'failures', 'absences']

@app.route('/predict', methods=['POST'])
def predict_course():
    try:
        # Parse JSON input
        input_data = request.get_json()

        # Extract user inputs
        age = input_data.get('age')
        studytime = input_data.get('studytime')  # Yes/No
        absences = input_data.get('absences')    # Rarely/Sometimes/Often
        failures = input_data.get('failures')    # None/Few/Many
              # Convert 'age' to an integer
        try:
            age = int(age)
        except ValueError:
            return jsonify({"error": "'age' must be a valid integer"}), 400
        # Map user inputs to numeric values
        studytime_mapped = 3 if studytime.lower() == "yes" else 1
        absences_mapped = {"rarely": 2, "sometimes": 6, "often": 12}.get(absences.lower(), 2)
        failures_mapped = {"none": 0, "few": 1, "many": 3}.get(failures.lower(), 0)

        # Add derived features
        effort_level = (studytime_mapped * 2) - failures_mapped
        reliability_score = 10 - absences_mapped - (failures_mapped * 2)
        course_category = "Advanced" if age >= 18 and studytime_mapped >= 3 else "Basic"

        # Create a DataFrame with the original + derived features
        input_df = pd.DataFrame([{
            "Age": age,
            "StudyTime": studytime_mapped,
            "Failures": failures_mapped,
            "Absences": absences_mapped
        }], columns=feature_columns)

        # Make the prediction
        prediction = model.predict(input_df)
        predicted_course = le.inverse_transform(prediction)[0]

        # Generate unique, user-friendly feedback
        feedback = f"Based on your age ({age}), effort level ({effort_level}), and reliability ({reliability_score}), the course '{predicted_course}' suits your learning style best!"

        # Return the predicted course and feedback
        return jsonify({
            "predicted_course": predicted_course,
            "feedback": feedback,
            "derived_metrics": {
                "effort_level": effort_level,
                "reliability_score": reliability_score,
                "course_category": course_category
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
