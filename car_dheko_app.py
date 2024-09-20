import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model, scaler, and encoder
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Load reference data
reference_data = pd.read_excel('FeaturesEngineered.xlsx')

# Define numerical and categorical columns
important_numerical_cols = ['kilometer', 'owner_no', 'model_year', 'registration_year', 'seats_count', 
                            'car_engine_cc', 'mileage', 'gear_box', 'age', 'mileage_per_Year']
important_categorical_cols = ['fuel_type', 'body_type', 'transmission_type', 'brand', 'model', 'color', 'location']

# Dynamic filtering based on manufacturer
def filter_data_by_brand(reference_data, brand):
    return reference_data[reference_data['brand'] == brand]

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter the car features below and get an estimated price.")

# Apply custom background
background_css = '''
    <style>
    .stApp {
        background-image:url("https://www.shutterstock.com/image-illustration/black-modern-car-headlights-front-600nw-771184300.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
'''
st.markdown(background_css, unsafe_allow_html=True)

# Sidebar for main selections
st.sidebar.title("Car Specifications")
brand = st.sidebar.selectbox("Brand", reference_data['brand'].unique().tolist())

# Filter data dynamically based on manufacturer
filtered_data = filter_data_by_brand(reference_data, brand)

# Update options based on filtered data
cities = filtered_data['location'].unique().tolist()
fuel_types = filtered_data['fuel_type'].unique().tolist()
body_types = filtered_data['body_type'].unique().tolist()
car_models = filtered_data['model'].unique().tolist()
colors = filtered_data['color'].unique().tolist()
transmission_types = filtered_data['transmission_type'].unique().tolist()

# Sidebar inputs
city = st.sidebar.selectbox("Location", cities)
fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types)
body_type = st.sidebar.selectbox("Body Type", body_types)
car_model = st.sidebar.selectbox("Car Model", car_models)
color = st.sidebar.selectbox("Color", colors)
transmission_type = st.sidebar.selectbox("Transmission Type", transmission_types)

# Main panel for numerical inputs
kilometers_driven = st.number_input("Kilometers Driven", min_value=0)
previous_owners = st.slider("Previous Owners", min_value=0, max_value=10)
model_year = st.slider("Model Year", min_value=2000, max_value=2023)
registration_year = st.slider("Registration Year", min_value=2000, max_value=2023)
seats = st.slider("Seats", min_value=2, max_value=8)
engine_capacity = st.number_input("Engine Capacity (cc)", min_value=500, max_value=5000)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=30.0)
gear_box = st.selectbox("Gear Box", [4, 5, 6])

# Calculate additional numerical features
car_age = model_year - registration_year
mileage_per_year = kilometers_driven / (car_age if car_age > 0 else 1)

# Combine all numerical data into a single array (make sure it has 10 features as expected by the scaler)
numerical_data = np.array([[kilometers_driven, previous_owners, model_year, registration_year, seats, 
                            engine_capacity, mileage, gear_box, car_age, mileage_per_year]])

# Scale the numerical data
scaled_numerical_data = scaler.transform(numerical_data)

# Collect categorical data into a DataFrame
categorical_data = pd.DataFrame([[fuel_type, body_type, transmission_type, brand, car_model, color, city]], 
                                columns=important_categorical_cols)

# Encode categorical data using the fitted encoder
encoded_categorical_data = encoder.transform(categorical_data).toarray()

# Combine numerical and encoded categorical data
final_data = np.hstack((scaled_numerical_data, encoded_categorical_data))

# Prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(final_data)
        price_in_lakhs = prediction[0] / 100000  # Convert to lakhs
        st.success(f"Predicted Car Price: ₹{price_in_lakhs:,.2f} Lakh")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
