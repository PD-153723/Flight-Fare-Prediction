import joblib
import pandas as pd
import streamlit as st

# Load the model
model = joblib.load(r"D:\desktop\CDAC_project\Project data\FPP_Final1.joblib")

# Streamlit app
st.title("Flight Price Prediction")

# Collecting form data
stops = st.number_input("Number of Stops", min_value=0, max_value=5, value=0, step=1)
class_type = st.selectbox("Class Type", options=[1, 2], format_func=lambda x: "Business" if x == 1 else "Economy")
duration = st.number_input("Duration (in hours)", min_value=0.0, value=1.0, step=0.1)
days_left = st.number_input("Days Left to Departure", min_value=0, value=30, step=1)

# Airline selection
airline = st.selectbox("Airline", options=['GO_FIRST', 'IndiGo', 'Air_India', 'SpiceJet', 'Vistara', 'AirAsia'])
airline_dict = {
    'GO_FIRST': [0, 0, 0, 1, 0, 0],
    'IndiGo': [0, 0, 1, 0, 0, 0],
    'Air_India': [0, 1, 0, 0, 0, 0],
    'SpiceJet': [0, 0, 0, 0, 0, 1],
    'Vistara': [1, 0, 0, 0, 0, 0],
    'AirAsia': [0, 0, 0, 0, 1, 0],
}
airline_encoded = airline_dict.get(airline, [0, 0, 0, 0, 0, 0])

# Source city selection
source = st.selectbox("Source City", options=['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Ahmedabad'])
source_dict = {
    'Delhi': [1, 0, 0, 0, 0, 0, 0],
    'Kolkata': [0, 0, 0, 1, 0, 0, 0],
    'Mumbai': [0, 1, 0, 0, 0, 0, 0],
    'Chennai': [0, 0, 0, 0, 0, 1, 0],
    'Bangalore': [0, 0, 1, 0, 0, 0, 0],
    'Hyderabad': [0, 0, 0, 0, 1, 0, 0],
    'Ahmedabad': [0, 0, 0, 0, 0, 0, 1],
}
source_encoded = source_dict.get(source, [0, 0, 0, 0, 0, 0, 0])

# Destination city selection
destination = st.selectbox("Destination City", options=['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Ahmedabad'])
destination_dict = {
    'Delhi': [1, 0, 0, 0, 0, 0, 0],
    'Kolkata': [0, 0, 0, 1, 0, 0, 0],
    'Mumbai': [0, 1, 0, 0, 0, 0, 0],
    'Chennai': [0, 0, 0, 0, 0, 1, 0],
    'Bangalore': [0, 0, 1, 0, 0, 0, 0],
    'Hyderabad': [0, 0, 0, 0, 1, 0, 0],
    'Ahmedabad': [0, 0, 0, 0, 0, 0, 1],
}
destination_encoded = destination_dict.get(destination, [0, 0, 0, 0, 0, 0, 0])

# Arrival time selection
arrival_time = st.selectbox("Arrival Time", options=['arrival_12_PM_6_PM', 'arrival_6_AM_12_PM', 'arrival_After_6_PM', 'arrival_Before_6_AM'])
arrival_dict = {
    'arrival_12_PM_6_PM': [1, 0, 0, 0],
    'arrival_6_AM_12_PM': [0, 1, 0, 0],
    'arrival_After_6_PM': [0, 0, 1, 0],
    'arrival_Before_6_AM': [0, 0, 0, 1],
}
arrival_encoded = arrival_dict.get(arrival_time, [0, 0, 0, 0])

# Departure time selection
departure_time = st.selectbox("Departure Time", options=['departure_12_PM_6_PM', 'departure_6_AM_12_PM', 'departure_After_6_PM', 'departure_Before_6_AM'])
departure_dict = {
    'departure_12_PM_6_PM': [1, 0, 0, 0],
    'departure_6_AM_12_PM': [0, 1, 0, 0],
    'departure_After_6_PM': [0, 0, 1, 0],
    'departure_Before_6_AM': [0, 0, 0, 1],
}
departure_encoded = departure_dict.get(departure_time, [0, 0, 0, 0])

# Prepare the feature vector for prediction
feature_vector = [
    stops,
    class_type,
    duration,
    days_left,
    *airline_encoded,
    *source_encoded,
    *destination_encoded,
    *arrival_encoded,
    *departure_encoded,
]

# Make the prediction
if st.button("Predict"):
    prediction = model.predict([feature_vector])
    output = round(prediction[0], 2)
    st.success(f"Your Flight price is Rs. {output}")
