import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------------------------------------
# Step 1: Load the Data
# ---------------------------------------------------------
# Assuming you have downloaded a dataset named 'crop_data.csv'
# It should have columns: N, P, K, temperature, humidity, ph, rainfall, label
try:
    df = pd.read_csv('crop_recommendation.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'crop_recommendation.csv' not found. Please download it and place it in the directory.")
    # Exiting here just for safety if the file isn't found
    exit()

# ---------------------------------------------------------
# Step 2: Preprocess the Data
# ---------------------------------------------------------
# Separate features (X) and target/labels (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label'] # The crop name

# Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# Step 3: Build and Train the Model
# ---------------------------------------------------------
# Initialize the Random Forest Classifier
# Random Forest works great for this as it handles non-linear data well
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 4: Evaluate the Model
# ---------------------------------------------------------
# Make predictions on the test set
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# Step 5: Save the Model for Future Use (Deployment)
# ---------------------------------------------------------
# Save the trained model to a file so you don't have to retrain it every time
joblib.dump(model, 'smart_agri_model.pkl')
print("\nModel saved as 'smart_agri_model.pkl'")

# ---------------------------------------------------------
# Step 6: Test with a Custom Input (Simulating the Adviser)
# ---------------------------------------------------------
def recommend_crop(n, p, k, temp, humidity, ph, rainfall):
    # Load the saved model
    loaded_model = joblib.load('smart_agri_model.pkl')
    
    # Create a numpy array for the input
    input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    
    # Make a prediction
    prediction = loaded_model.predict(input_data)
    return prediction[0]

# Example usage:
print("\n--- Smart Agriculture Adviser Test ---")
# Let's say a farmer inputs these specific soil and weather conditions:
sample_N = 90
sample_P = 42
sample_K = 43
sample_temp = 20.8
sample_humidity = 82.0
sample_ph = 6.5
sample_rainfall = 202.9

recommended_crop = recommend_crop(sample_N, sample_P, sample_K, sample_temp, sample_humidity, sample_ph, sample_rainfall)
print(f"Based on the conditions, the recommended crop is: **{recommended_crop.capitalize()}**")

import streamlit as st
import numpy as np
import joblib

# Load the trained model
# Make sure 'smart_agri_model.pkl' is in the same directory
try:
    model = joblib.load('smart_agri_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please train and save the model first.")
    st.stop()

# Build the Web App UI
st.title("🌱 Smart Agriculture Adviser")
st.write("Enter your soil and weather conditions to get the best crop recommendation.")

# Create input fields for the user
col1, col2 = st.columns(2)

with col1:
    n = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=90)
    p = st.number_input("Phosphorous (P)", min_value=0, max_value=150, value=42)
    k = st.number_input("Potassium (K)", min_value=0, max_value=205, value=43)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.8)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=202.9)

# Create a predict button
if st.button("Recommend Crop"):
    # Prepare the input for the model
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    
    # Get prediction
    prediction = model.predict(input_data)[0]
    
    # Display the result
    st.success(f"Based on the conditions, you should plant: **{prediction.capitalize()}** 🌾")
    pd.read_csv('crop_recommendation.csv')