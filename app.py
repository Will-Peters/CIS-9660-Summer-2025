import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk

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
    "Entire home/apt": ["Boat","Entire condo","Entire guest suite","Entire guesthouse","Entire home","Entire loft","Entire place","Entire rental unit","Entire serviced apartment","Entire townhouse"],
    "Private room": ["Private room","Private room in bed and breakfast","Private room in condo","Private room in guest suite","Private room in guesthouse","Private room in home","Private room in loft","Private room in rental unit","Private room in townhouse"],
    "Hotel room": ["Room in aparthotel","Room in boutique hotel"],
    "Shared room": ["Shared room in home","Shared room in loft","Shared room in rental unit"]
}


# User input form

Neighbourhood = st.selectbox("Borough", list(borough_to_neighborhoods.keys()))
Neighbourhood_cleansed = st.selectbox("Neighborhood", borough_to_neighborhoods[Neighbourhood])  
room_type = st.selectbox("Room Group", list(Property_Group_to_Type.keys()))
room_detail = st.selectbox("Room detail", Property_Group_to_Type[room_type])

with st.form("user_inputs"):
    bedrooms = st.slider("Bedrooms", 0, 5, 1)
    beds = st.slider("Beds", 0, 5, 1)
    superhost = st.selectbox("Is the host a Superhost?", ["Yes", "No"])
    drive_duration = st.number_input("Drive Duration to Times Square (minutes)", 0, 120, 20) * 60
    drive_distance_km = st.number_input("Drive Distance to Times Square (kilometers)", 0, 20, 2)
    transit_duration = st.number_input("Transit Duration to Times Square (minutes)", 0, 120, 20) * 60
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame({
        "bedrooms": [bedrooms],
        "beds": [beds],
        "room_type": [room_type],
        "host_is_superhost": [superhost],
        "neighbourhood_group_cleansed": [Neighbourhood],
        "drive_duration": [drive_duration],
        "drive_distance_km": [drive_distance_km],
        "transit_duration": [transit_duration],
        "property_type": [room_detail],
        "neighbourhood_cleansed": [Neighbourhood_cleansed],
        "review_scores_rating": [4.7],
        "review_scores_accuracy": [4.5],
        "review_scores_cleanliness": [4.6],
        "review_scores_checkin": [4.8],
        "review_scores_communication": [4.9],
        "review_scores_location": [4.6],
        "review_scores_value": [4.4]
    })

    log_price_pred = model.predict(input_df)[0]
    prediction = np.expm1(log_price_pred)
    st.success(f"Estimated Rental Price: ${prediction * 0.9:.2f} – ${prediction * 1.1:.2f}")
st.markdown("____________________________________________________________________________________________________")
st.markdown("### 📊 Map Visualization")
st.markdown("____________________________________________________________________________________________________")
df_cleaned = pd.read_pickle("enriched_df.pkl")

selected_beds = st.slider("Filter by number of beds", min_value=0, max_value=5, value=1)

# Filter and aggregate
filtered_df = df_cleaned[df_cleaned['beds'] == selected_beds]

avg_price_df = (
    filtered_df.
    groupby('neighbourhood_cleansed', as_index=False)
    .agg(
         avg_price=('price', 'mean'),
         lat=('latitude', 'mean'),
         lon=('longitude', 'mean')
    )
)

# Create pydeck map

text_layer = pdk.Layer(
    "TextLayer",
    data=avg_price_df,
    get_position='[lon, lat]',
    get_text='avg_price',
    get_size=16,
    get_color=[0, 0, 0],  # black text
    get_angle=0,
    pickable=False
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=avg_price_df,
    get_position='[lon, lat]',
    get_color='[255, 0, 0, 160]',
    get_radius="avg_price * 10",  # Scale bubbles by price
    pickable=True
)

view_state = pdk.ViewState(
    latitude=40.7128,
    longitude=-74.0060,
    zoom=10,
    pitch=0
)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/streets-v11',
    initial_view_state=view_state,
    layers=[layer],
    tooltip={"text": "{neighbourhood_cleansed}\nAvg Price: ${avg_price}"}
))

