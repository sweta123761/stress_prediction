import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset (Replace 'Stress-Lysis.csv' with the actual file path)
stress_data_url = 'Stress-Lysis.csv'  # Change this to your dataset file

df = pd.read_csv(stress_data_url)

# Selecting relevant features
X = df[['Humidity', 'Temperature', 'Step count']]  # Features
y = df['Stress Level']  # Target Variable

# Splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to predict stress level for new input values
def predict_stress(humidity, temperature, step_count):
    input_data = np.array([[humidity, temperature, step_count]])
    input_data_scaled = scaler.transform(input_data)
    predicted_stress = model.predict(input_data_scaled)
    return int(predicted_stress[0])

# Example usage
humidity = float(input("Enter Humidity: "))
temperature = float(input("Enter Temperature: "))
step_count = float(input("Enter Step Count: "))
predicted_stress = predict_stress(humidity, temperature, step_count)
print(f"Predicted Stress Level: {predicted_stress}")
import joblib

# Save the trained model
joblib.dump(model, "stress_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")

