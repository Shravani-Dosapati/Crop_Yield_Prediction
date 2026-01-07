import streamlit as st
import pandas as pd
import pickle
import sklearn

with open("grid_ridge.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Crop Yield Prediction")
st.write("Enter the details below to predict the crop yield (in tons per hectare):")

region = st.selectbox("Region", ["North", "South", "East", "West"])
soil_type = st.selectbox("Soil Type", ["Sandy", "Clay", "Silty", "Peaty", "Chalky", "Loamy"])
crop = st.selectbox("Crop", ["Wheat", "Rice", "Maize", "Barley", "Soybean"])
rainfall = st.number_input("Rainfall (mm)", min_value=100, max_value=1000, value=100, step=1)
temperature = st.number_input("Temperature (Celsius)", min_value=0, max_value=40, value=25, step=1)
Fertilizer_Used = st.number_input("Fertilizer Used (kg/ha)", min_value=60, max_value=150, value=60, step=1)
Irrigation_Used = st.number_input("Irrigation Used (mm)", min_value=0, max_value=5000, value=0, step=1)
Weather_Condition = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Windy", "Humid"])
Days_to_Harvest = st.number_input("Days to Harvest", min_value=60, max_value=365, value=60, step=2)

# IMPORTANT: Must exactly match the feature names used in training
feature_names = [
    'Region',
    'Soil_Type',
    'Crop',
    'Rainfall_mm',
    'Temperature_Celsius',
    'Fertilizer_Used',
    'Irrigation_Used',
    'Weather_Condition',
    'Days_to_Harvest'
]

if st.button("Predict Yield"):

    input_df = pd.DataFrame([{
        'Region': region,
        'Soil_Type': soil_type,
        'Crop': crop,
        'Rainfall_mm': rainfall,
        'Temperature_Celsius': temperature,
        'Fertilizer_Used': Fertilizer_Used,
        'Irrigation_Used': Irrigation_Used,
        'Weather_Condition': Weather_Condition,
        'Days_to_Harvest': Days_to_Harvest
    }], columns=feature_names)

    prediction = model.predict(input_df)[0]

    st.success(f"The predicted crop yield is {prediction:.2f} tons per hectare.")
