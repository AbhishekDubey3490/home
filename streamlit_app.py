import streamlit as st
import pickle
import numpy as np

# Load model
with open('house_price_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🏡 House Price Prediction App")

# Input fields
size = st.number_input("Enter House Size (sqft)")
age = st.number_input("Enter Age of House")
crime_rate = st.number_input("Enter Crime Rate (per 1000 people)")
distance = st.number_input("Enter Distance to City Center (miles)")

if st.button("Predict Price"):
    features = np.array([size, age, crime_rate, distance]).reshape(1, -1)
    prediction = model.predict(features)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
