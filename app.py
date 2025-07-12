import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import os
os.environ["MAPBOX_API_KEY"] = "pk.eyJ1Ijoid2lsbGlhbXAzMSIsImEiOiJjbWNxc2w5Mmcwa2tyMmpxMTB3aGxnOHg1In0.jHfWqLwGh_sFRuObzNtA1g"
from PIL import Image
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load model and preprocessor
rent_model = joblib.load("Regression_model.pkl")
rent_preprocessor = joblib.load("preprocessor.pkl")
attrition_model = joblib.load("attrition_model.pkl")
attrition_preprocessor = joblib.load("attrition_preprocessor.pkl")

st.set_page_config(page_title="AI Decision App", layout="wide")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["AirBNB Finder", "AirBNB Finder Metrics", "Employee Attrition Predictor","Employee Attrition Metrics/Analysis"])

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
        accommodates = st.slider("Number of People", 0, 10, 1)
        Availability = st.slider("Number of days available", 0, 365, 1)
        superhost = st.selectbox("Is the host a Superhost?", ["Yes", "No"])
        bathrooms = st.slider("Bathrooms", 0, 5, 1)
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
            "review_scores_value": [4.4],
            "bathrooms": [bathrooms],
            "accommodates": [accommodates],
            "availability_365_%": [Availability],
        })

        log_price_pred = rent_model.predict(input_df)[0]
        prediction = np.expm1(log_price_pred)
        st.success(f"Estimated Rental Price: ${prediction:.2f}")
        beginning = prediction*0.9
        end = prediction*1.1
        st.success(f"Range: {beginning:.2f} - {end:.2f}")

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
    xgb_model = joblib.load("Regression_model.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")

    # Predict
    y_pred = xgb_model.predict(X_test)
    residuals = y_test - y_pred

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    std_resid = np.std(residuals)

    # Show metrics
    st.metric("R²", f"{r2:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("Residual Std. Dev", f"{std_resid:.2f}")

    residuals = y_test - y_pred
    bins = pd.cut(y_pred, bins=np.linspace(min(y_pred), max(y_pred), 10))
    std_by_bin = residuals.groupby(bins).std()

    fig, ax = plt.subplots()
    ax.hist(y_pred, bins=10, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.set_title("Frequency of Predictions by Price Bin")
    ax.set_xlabel("Predicted Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
        
with tab3:
    st.title("🧑‍💼 Employee Attrition Classifier")
    st.markdown("Predict whether an employee is likely to leave the company.")

    # Example form inputs (adjust based on your dataset)
    with st.form("attrition_form"): 
        age = st.slider("Age", 18, 60, 30)
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        distance_from_home = st.slider("Distance from Home (miles)", 0, 30, 5)
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
        percent_salary_hike = st.slider("Percent Salary Hike", 10, 25, 1)
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

        st.success(f"Prediction: {'Leave' if prediction == 1 else 'Stay'}")
        st.success(f"Probability of leaving: {probability:.0%}")

with tab4:
    st.subheader("Confusion Matrix")
    conf_matrix_img = Image.open("confusion_matrix.png")
    st.image(conf_matrix_img)

    st.subheader("Precision-Recall Curve")
    pr_img = Image.open("pr_curve.png")
    st.image(pr_img)

    st.subheader("SHAP Summary")
    shap_img = Image.open("shap_summary_plot.png")
    st.image(shap_img)

    with open("classification_metrics.json", "r") as f:
        metrics = json.load(f)
    st.write(metrics)

    st.header("Classification Report")
    st.write("**ROC AUC:**", round(metrics['roc_auc'], 4))

    for label in ['0', '1', 'macro avg', 'weighted avg']:
        st.subheader(f"Class {label}")
        st.write(f"Precision: {metrics[label]['precision']:.2f}")
        st.write(f"Recall: {metrics[label]['recall']:.2f}")
        st.write(f"F1-score: {metrics[label]['f1-score']:.2f}")
        st.write(f"Support: {metrics[label]['support']}")
