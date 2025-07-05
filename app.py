import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="SmartRent Finder", layout="wide")
st.title("🌉 NYC Rent Price Predictor")
st.markdown("""
Enter apartment features to estimate rental price and evaluate commute time to Times Square.
""")

borough_to_neighborhoods = {
    "Manhattan": ["Battery Park City", "Central Park", "Chelsea", "Chinatown", "East Harlem", "East Village", "Financial District", "Greenwich Village", "Harlem", "Hell's Kitchen", "Inwood", "Lower East Side", "Morningside Heights", "SoHo", "Tribeca", "Upper East Side", "Upper West Side", "Washington Heights", "West Village"],
    "Brooklyn": ["Bedford-Stuyvesant", "Boerum Hill", "Brooklyn Heights", "Bushwick", "Canarsie", "Carroll Gardens", "Clinton Hill", "Cobble Hill", "Crown Heights", "Cypress Hills", "Downtown Brooklyn", "DUMBO", "East Flatbush", "Flatbush", "Flatlands", "Fort Greene", "Gowanus", "Gravesend", "Greenpoint", "Park Slope", "Prospect Heights", "Prospect-Lefferts Gardens", "Red Hook", "South Slope", "Sunset Park", "Williamsburg", "Windsor Terrace"],
    "Queens": ["Astoria", "Bayside", "Ditmars Steinway", "East Elmhurst", "Elmhurst", "Flushing", "Forest Hills", "Jackson Heights", "Jamaica", "Kew Gardens", "Long Island City", "Queens Village", "Rego Park", "Richmond Hill", "Ridgewood", "Rockaway Beach", "Sunnyside", "Woodhaven", "Woodside"],
    "Bronx": ["Allerton", "Bronx Park", "Bronxdale", "City Island", "Concourse", "Eastchester", "Fordham", "Kingsbridge", "Melrose", "Mott Haven", "Parkchester", "Riverdale", "Soundview", "Spuyten Duyvil", "University Heights", "Wakefield", "Williamsbridge"],
    "Staten Island": ["Arrochar", "Concord", "New Springville", "Shore Acres", "St. George", "Tompkinsville"]
}

Property_Group_to_Type = {
    "Entire Space": ["Boat","Entire condo","Entire guest suite","Entire guesthouse","Entire home","Entire loft","Entire place","Entire rental unit","Entire serviced apartment","Entire townhouse"],
    "Private Space": ["Private room","Private room in bed and breakfast","Private room in condo","Private room in guest suite","Private room in guesthouse","Private room in home","Private room in loft","Private room in rental unit","Private room in townhouse"],
    "Room": ["Room in aparthotel","Room in boutique hotel"],
    "Shared Room": ["Shared room in home","Shared room in loft","Shared room in rental unit"]
}


# User input form
with st.form("user_inputs"):
    bedrooms = st.slider("Bedrooms", 0, 5, 1)
    beds = st.slider("Beds", 0, 5, 1)
    superhost = st.selectbox("Is the host a Superhost?", ["Yes", "No"])
    Neighbourhood = st.selectbox("Borough", list(borough_to_neighborhoods.keys()))
    Neighbourhood_cleansed = st.selectbox("Neighborhood", borough_to_neighborhoods[Neighbourhood])    
    drive_duration = st.number_input("Drive Duration to Times Square (minutes)", 0, 120, 20) * 60
    drive_distance_km = st.number_input("Drive Distance to Times Square (kilometers)", 0, 20, 2)
    transit_duration = st.number_input("Transit Duration to Times Square (minutes)", 0, 120, 20) * 60
    property_group = st.selectbox("Property Group", list(Property_Group_to_Type.keys()))
    property_type = st.selectbox("Property Type", Property_Group_to_Type[property_group])
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame({
        "bedrooms": [bedrooms],
        "beds": [beds],
        "room_type": [room_type],
        "host_is_superhost": [superhost],
        "neighbourhood_group_cleansed": [neighbourhood],
        "drive_duration": [drive_duration],
        "drive_distance_km": [drive_distance_km],
        "transit_duration": [transit_duration],
        "property_type": [property_type],
        "neighbourhood_cleansed": [neighbourhood_cleansed],
        "review_scores_rating": [4.7],
        "review_scores_accuracy": [4.5],
        "review_scores_cleanliness": [4.6],
        "review_scores_checkin": [4.8],
        "review_scores_communication": [4.9],
        "review_scores_location": [4.6],
        "review_scores_value": [4.4]
    })

    input_transformed = preprocessor.transform(input_df)
    prediction = model.predict(input_transformed)[0]
    st.success(f"Estimated Rental Price: ${prediction:.2f}")
