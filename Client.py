import requests

# Define the API URL
api_url = 'http://localhost:5000/predict'  # Adjust the URL if necessary

# User input data
user_data = {
    'age': 30,
    'isParent': True,
    'childAge': 5
}

# Send POST request to the API
response = requests.post(api_url, json=user_data,verify=False)

# Handle the response
if response.status_code == 200:
    result = response.json()  # Get prediction result from API
    print(f"Predicted Course: {result}")
else:
    print(f"Error: {response.status_code}, {response.text}")
