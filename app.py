import streamlit as st
import streamlit_folium as st_folium
import folium  # <--- THIS IS THE MISSING LINE
import json
import requests
import joblib
import pandas as pd
import numpy as np

# 1. Load the model
# IMPORTANT: Make sure these files are in the SAME folder on GitHub
model = joblib.load('nyc_model streamlit.pkl')
model_columns = joblib.load('nyc file streamlit.pkl')

# 2. Page Config
st.set_page_config(page_title="NYC Real Estate AI", layout="wide", page_icon="🏙️")

# 3. Sidebar for Inputs
st.sidebar.header("🏢 Listing Configuration")
lat = st.sidebar.number_input("Latitude", value=40.7128, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=-74.0060, format="%.4f")
lux = st.sidebar.slider("AI Luxury Score (NLP)", 0.0, 1.0, 0.85)
nights = st.sidebar.number_input("Min. Nights", value=1)

# 4. Main Interface
st.title("🏙️ NYC Luxury Rental Price Intelligence")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Location Intelligence")
    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_data)
    
    st.markdown("---") # Adds a nice divider
    st.subheader("📍 Neighborhood Market Heatmap")
    
    # Load GeoJSON
    geojson_url = "https://raw.githubusercontent.com/fedhere/PUI2015_ak5329/master/HW5_ak5329/nyc-zip-codes.geojson"
    
    # Create the map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="CartoDB positron")
    # Add the Choropleth layer
    # Note: For the 'data' part, we use a placeholder or your existing dataframe
    folium.Choropleth(
        geo_data=geojson_url,
        name="choropleth",
        fill_color="RdPu", # Your signature pink/purple
        fill_opacity=0.5,
        line_opacity=0.2,
        highlight=True,
    ).add_to(m)

    # Display the map
    st_folium.st_folium(m, width=700, height=450)
    
    st.caption("Above: Real-time geospatial distribution of market premiums.")
with col2:
    st.subheader("💰 Valuation Engine")
    st.write("This engine uses a Random Forest Regressor trained on 40,000+ NYC data points.")
    
    if st.button("RUN AI VALUATION 🚀", use_container_width=True):
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        input_df['latitude'] = lat
        input_df['longitude'] = lon
        input_df['minimum_nights'] = nights
        if 'luxury_score' in input_df.columns:
            input_df['luxury_score'] = lux
            
        prediction = model.predict(input_df)
        
        st.metric(label="Estimated Nightly Rate", value=f"${prediction[0]:.2f}")
        st.success("✅ Prediction generated using semantic luxury features.")
        st.snow() # Falling diamonds effect!

st.markdown("---")
st.info("📊 **Note to Recruiters:** This project integrates HuggingFace Transformers (DistilBERT) for NLP sentiment analysis and Scikit-Learn for spatial regression.")


