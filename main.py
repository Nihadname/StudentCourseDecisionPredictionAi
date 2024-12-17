import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib
data = pd.read_csv("student-mat.csv", sep=";")

# Display the first few rows
print(data.head())
data['course_preference'] = data['G3'].apply(lambda x: 'Advanced' if x > 15 else 'Basic')

# Features (X) and target (y)
X = data[['age', 'studytime', 'failures', 'absences']]
y = data['course_preference']  # Target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model and label encoder
joblib.dump(model, "course_predictor_model.pkl")
joblib.dump(le, "label_encoder.pkl")
