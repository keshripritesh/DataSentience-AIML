import streamlit as st
import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np


model = joblib.load('random_forest_model.pkl')


st.header("GDP Predictor🌏",divider =True)
st.write("This model predicts a Country's Per Capita GDP based on suitable factors.  \nThe dataset used for this project can be found on kaggle : (https://www.kaggle.com/rutikbhoyar/gdp-prediction-dataset)")


region_columns = [
    "Region_ASIA (EX. NEAR EAST)         ",
    "Region_BALTICS                            ",
    "Region_C.W. OF IND. STATES ",
    "Region_EASTERN EUROPE                     ",
    "Region_LATIN AMER. & CARIB    ",
    "Region_NEAR EAST                          ",
    "Region_NORTHERN AFRICA                    ",
    "Region_NORTHERN AMERICA                   ",
    "Region_OCEANIA                            ",
    "Region_SUB-SAHARAN AFRICA                 ",
    "Region_WESTERN EUROPE                     "
]

region_choice = st.selectbox("Region",("ASIA (EX. NEAR EAST)","BALTICS","C.W. OF IND. STATES","EASTERN EUROPE","LATIN AMER. & CARIB","NEAR EAST","NORTHERN AFRICA","NORTHERN AMERICA","OCEANIA","SUB-SAHARAN AFRICA","WESTERN EUROPE",))
region_data = pd.DataFrame(np.zeros((1, len(region_columns))), columns=region_columns)
region_data[f"Region_{region_choice}"] = 1


climate_map = {
    "Desert / Hot": 1,
    "Hot & Tropical": 1.5,
    "Tropical": 2,
    "Cold & Tropical": 2.5,
    "Cold": 3,
    "Cold (Category 4)": 4
}

climate_choice = st.selectbox("Select Climate Type", list(climate_map.keys()))
climate_value = climate_map[climate_choice]

col1,col2,col3 = st.columns(3)

with col1:
    population = st.number_input("Population",min_value=5000,step = 1000,value = 30000000)
    net_migration = st.number_input("Net migration ratio",min_value=-500.0,value=0.04)
    phones = st.number_input("Phones per 1000",value=233)
    agriculture = st.number_input("Agriculture sector %age of economy",step = 0.001)
    

with col2:
    area = st.number_input("Area(per sq. mt.)",value = 600000.00)
    mortality = st.number_input("Infant mortality per 1000 births",value=35)
    birth_rate = st.number_input("Birth Rate per 1000",value=22)
    industry = st.number_input("Industry sector %age of economy",step=0.001)
    

with col3: 
    coastline = st.number_input("Coastline (coast/area ratio)",value = 21.00)
    literacy = st.number_input("Literacy %",value=77)
    death_rate = st.number_input("Death rate per 1000",value=9)
    service = st.number_input("Service Sector %age of economy")
    

new_col1,new_col2 = st.columns(2)

with new_col1:
        arable = st.slider("Arable land %",min_value =0.00,max_value=100.00,value=14.00,step=0.01)

with new_col2:
        crop = st.slider("Crop land %",min_value =0.00,max_value=100.00,value=14.00,step=0.01)

other = 100-(crop+arable)

pop_density = population/area


numeric_input_df = pd.DataFrame([{
        'Population': population,
        'Area (sq. mi.)': area,
        'Pop. Density (per sq. mi.)': pop_density,
        'Coastline (coast/area ratio)': coastline,
        'Net migration': net_migration,
        'Infant mortality (per 1000 births)': mortality,
        'Literacy (%)': literacy,
        'Phones (per 1000)':phones ,
        'Arable (%)': arable,
        'Crops (%)': crop,
        'Other (%)': other,
        'Climate':climate_value ,
        'Birthrate':birth_rate ,
        'Deathrate': death_rate,
        'Agriculture':agriculture,
        'Industry':industry ,
        'Service': service

}])




input_df = pd.concat([numeric_input_df,region_data],axis=1)


# Load the correct feature list used during training
expected_features = model.feature_names_in_

# Reorder and align input_df to match training exactly
input_df = input_df.reindex(columns=expected_features, fill_value=0)


st.markdown("###")

if st.button("Predict GDP💰"):
    prediction = round(model.predict(input_df)[0],2)
    st.success(f"The predicted gdp is : ${prediction}")

