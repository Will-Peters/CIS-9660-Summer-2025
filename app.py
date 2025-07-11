import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import os
os.environ["MAPBOX_API_KEY"] = "pk.eyJ1Ijoid2lsbGlhbXAzMSIsImEiOiJjbWNxc2w5Mmcwa2tyMmpxMTB3aGxnOHg1In0.jHfWqLwGh_sFRuObzNtA1g"

# Load model and preprocessor
rent_model = joblib.load("Regression_model.pkl")
rent_preprocessor = joblib.load("preprocessor.pkl")
attrition_model = joblib.load("attrition_model.pkl")
attrition_preprocessor = joblib.load("attrition_preprocessor.pkl")

st.set_page_config(page_title="AI Decision App", layout="wide")

# Tabs
tab1, tab2 = st.tabs(["🏠 SmartRent Finder", "🧑‍💼 Employee Attrition Predictor"])

with tab1:
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
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame({
            "bedrooms": [bedrooms],
            "beds": [beds],
            "room_type": [room_type],
            "host_is_superhost": [superhost],
            "neighbourhood_group_cleansed": [Neighbourhood],
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

        log_price_pred = rent_model.predict(input_df)[0]
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

    avg_price_df["avg_price"] = avg_price_df["avg_price"].round(0)

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
        get_radius="avg_price * 2",  # Scale bubbles by price
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

with tab2:
    st.title("🧑‍💼 Employee Attrition Classifier")
    st.markdown("Predict whether an employee is likely to leave the company.")

    # Example form inputs (adjust based on your dataset)
    with st.form("attrition_form"): 
        age = st.slider("Age", 18, 60, 30)
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        distance_from_home = st.slider("Distance from Home (miles)", 0, 50, 10)
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5])  # 1='Below College', 5='Doctor'
        gender = st.selectbox("Gender", ["Male", "Female"])
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
            "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"
        ])
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
        num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 1)
        percent_salary_hike = st.slider("Percent Salary Hike", 10, 25, 15)
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_current_role = st.slider("Years in Current Role", 0, 20, 4)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        years_with_curr_manager = st.slider("Years with Current Manager", 0, 17, 3)

        submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        input_df = pd.DataFrame({
            "Age": [age],
            "BusinessTravel": [business_travel],
            "DistanceFromHome": [distance_from_home],
            "Education": [education],
            "Gender": [gender],
            "JobInvolvement": [job_involvement],
            "JobLevel": [job_level],
            "JobRole": [job_role],
            "JobSatisfaction": [job_satisfaction],
            "MaritalStatus": [marital_status],
            "MonthlyIncome": [monthly_income],
            "NumCompaniesWorked": [num_companies_worked],
            "PercentSalaryHike": [percent_salary_hike],
            "PerformanceRating": [performance_rating],
            "RelationshipSatisfaction": [relationship_satisfaction],
            "WorkLifeBalance": [work_life_balance],
            "YearsAtCompany": [years_at_company],
            "YearsInCurrentRole": [years_in_current_role],
            "YearsSinceLastPromotion": [years_since_last_promotion],
            "YearsWithCurrManager": [years_with_curr_manager]
        })

        input_processed = attrition_preprocessor.transform(input_df)
        prediction = attrition_model.predict(input_processed)[0]
        probability = attrition_model.predict_proba(input_processed)[0][1]

        st.success(f"Prediction: {'Leave' if prediction == 1 else 'Stay'} (Prob: {probability:.2f})")