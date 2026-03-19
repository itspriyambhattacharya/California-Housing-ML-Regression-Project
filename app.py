import streamlit as st
import joblib
import pandas as pd

st.title("California House Price Prediction")
st.markdown("---")

# load model


@st.cache_resource
def load_model():
    pipeline = joblib.load('pipeline.pkl')
    col_names = joblib.load('col_names.pkl')
    return pipeline, col_names


pipeline, col_names = load_model()

st.sidebar.write("Model Used: Random Forest")

# ui design
lon = st.number_input(
    "Longitude:"
)

lat = st.number_input(
    "Latitude:"
)

housing_median_age = st.number_input(
    "House Median Age:"
)

households = st.number_input(
    "Households"
)

median_income = st.number_input(
    "Median Income"
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ['NEAR BAY', 'NEAR OCEAN', 'ISLAND', 'INLAND', '<1H OCEAN']
)

rooms_per_household = st.number_input(
    "Rooms per Household"
)

population_per_household = st.number_input(
    "Population per household"
)

bedroom_per_room = st.number_input(
    "Bedromm per household"
)

if st.button("Predict"):
    with st.spinner("Predicting Price"):
        ns = pd.DataFrame({
            'longitude': [lon],
            'latitude': [lat],
            'housing_median_age': [housing_median_age],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity],
            'rooms_per_household': [rooms_per_household],
            'population_per_household': [population_per_household],
            'bedroom_per_room': [bedroom_per_room]
        })
        output = pipeline.predict(ns)
        st.success(f"The Predicted Price is: {output[0]}")

st.markdown("---")
st.write("Developed by Priyam Bhattacharya")
