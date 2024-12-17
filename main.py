import joblib
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
api_url = 'http://localhost:5066/api/RequstToRegister'  # Adjust URL based on the port you're using
response = requests.get(api_url,verify=False)

# Initialize df to avoid potential error if the API call fails
df = pd.DataFrame()

if response.status_code == 200:
    print('Response data:', response.json())
    # Convert the JSON data into a DataFrame
    df = pd.DataFrame(response.json())
else:
    print(f'Failed to connect, Status code: {response.status_code}')
from tabulate import tabulate

# Pretty print the DataFrame
print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
# Feature selection
# Feature selection and target encoding
X = df[['age', 'isParent', 'childAge']]  # Example features
y = df['choosenCourse']  # Target variable

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, 'course_predictor_model.pkl')
joblib.dump(le, 'label_encoder.pkl')