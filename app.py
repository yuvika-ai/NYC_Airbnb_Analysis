import streamlit as st
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
    
    # --- CHOROPLETH SECTION ---
    # When you are ready with the 'nyc_map_merged' data, we will put it here.
    # For the Tata application, the 'st.map' above is enough to prove 'Geospatial' skills!
    st.info("🗺️ *Interactive Neighborhood Heatmap integration in progress...*")

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