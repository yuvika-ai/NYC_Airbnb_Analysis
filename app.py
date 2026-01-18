import streamlit as st
import pydeck as pdk
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# 1. Page Config (MUST BE FIRST)
# -----------------------------
st.set_page_config(
    page_title="NYC Real Estate AI",
    layout="wide",
    page_icon="🏙️"
)

# -----------------------------
# 2. Luxe Dark Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b0014, #12001f);
    color: #ffd6f5;
}
h1, h2, h3 {
    color: #ff9fe5 !important;
}
[data-testid="stMetricValue"] {
    color: #9efff7 !important;
}
.stButton>button {
    background-color: #ff9fe5;
    color: black;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. Load Data & Model
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("nyc_model streamlit.pkl")
    columns = joblib.load("nyc file streamlit.pkl")
    return model, columns

@st.cache_data
def load_heatmap_data():
    df = pd.read_csv("neighbourhood_prices.csv")
    # Fix potential column name issues (Capitalization)
    df.columns = df.columns.str.lower()
    return df

try:
    model, model_columns = load_assets()
    nyc_df = load_heatmap_data()
except Exception as e:
    st.error(f"Error loading assets: {e}")

# -----------------------------
# 4. Sidebar Configuration
# -----------------------------
st.sidebar.header("🏢 Listing Configuration")
lat_input = st.sidebar.number_input("Latitude", value=40.7128, format="%.4f")
lon_input = st.sidebar.number_input("Longitude", value=-74.0060, format="%.4f")
lux = st.sidebar.slider("AI Luxury Score (NLP)", 0.0, 1.0, 0.85)
nights = st.sidebar.number_input("Min. Nights", value=1, min_value=1)

# -----------------------------
# 5. Main UI
# -----------------------------
st.title("🏙️ NYC Luxury Rental Price Intelligence")
st.markdown("---")

col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.subheader("🗺️ NYC Price Density Map")
    
    # Layer 1: Heatmap (The Glow)
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=nyc_df,
        get_position=["longitude", "latitude"],
        get_weight="price",
        radiusPixels=60,
        intensity=15,
        threshold=0.05,
        colorRange=[
            [254, 235, 226, 100],
            [251, 180, 185, 150],
            [247, 104, 161, 200],
            [197, 27, 138, 220],
            [122, 1, 119, 255]
        ]
    )

    view_state = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=10,
        pitch=45
    )

    r = pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        # This style does NOT require a Mapbox token to show the streets
        map_style="light", 
    )
    
    st.pydeck_chart(r)

with col2:
    st.subheader("💰 Valuation Engine")
    st.write("Random Forest Regressor: 40k+ data points.")
    
    if st.button("RUN AI VALUATION 🚀", use_container_width=True):
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        input_df["latitude"] = lat_input
        input_df["longitude"] = lon_input
        input_df["minimum_nights"] = nights
        
        if "luxury_score" in input_df.columns:
            input_df["luxury_score"] = lux
            
        prediction = model.predict(input_df)
        
        st.metric(label="Estimated Nightly Rate", value=f"${prediction[0]:.2f}")
        st.success("✅ Prediction generated.")
        st.snow()

st.markdown("---")
st.info("📊 **Stack:** Scikit-learn, PyDeck Geospatial, Streamlit Cloud, NLP Signals.")

