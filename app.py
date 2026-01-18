import streamlit as st
import pydeck as pdk
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Page Config (FIRST)
# -----------------------------
st.set_page_config(
    page_title="NYC Real Estate AI",
    layout="wide",
    page_icon="🏙️"
)

# -----------------------------
# Styling (dark, luxe, poppy)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b0014, #12001f);
    color: #ffd6f5;
}
h1, h2, h3 {
    color: #ff9fe5;
}
[data-testid="stMetricValue"] {
    color: #9efff7;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("nyc_model streamlit.pkl")
    columns = joblib.load("nyc file streamlit.pkl")
    return model, columns

model, model_columns = load_model()

# -----------------------------
# Load heatmap data (CSV ONLY)
# -----------------------------
@st.cache_data
def load_heatmap_data():
    return pd.read_csv("neighbourhood_prices.csv")

nyc_df = load_heatmap_data()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🏢 Listing Configuration")

lat = st.sidebar.number_input("Latitude", value=40.7128, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=-74.0060, format="%.4f")
lux = st.sidebar.slider("AI Luxury Score (NLP)", 0.0, 1.0, 0.85)
nights = st.sidebar.number_input("Min. Nights", value=1, min_value=1)

# -----------------------------
# Main Title
# -----------------------------
st.title("🏙️ NYC Luxury Rental Price Intelligence")
st.markdown("---")

col1, col2 = st.columns([1, 1])

# -----------------------------
# MAP
# -----------------------------
with col1:
    st.subheader("🗺️ NYC Price Density Map")

    # CLEAN DATA BEFORE MAPPING
    # This ensures PyDeck doesn't get confused by "string" numbers
    nyc_df['latitude'] = pd.to_numeric(nyc_df['latitude'], errors='coerce')
    nyc_df['longitude'] = pd.to_numeric(nyc_df['longitude'], errors='coerce')
    nyc_df = nyc_df.dropna(subset=['latitude', 'longitude'])

    # THE HEATMAP (The Glow)
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=nyc_df,
        get_position=["longitude", "latitude"],
        get_weight="price",
        radiusPixels=60,
        intensity=25, 
        threshold=0.01,
        colorRange=[
            [254, 235, 226, 100],
            [251, 180, 185, 150],
            [247, 104, 161, 200],
            [197, 27, 138, 220],
            [122, 1, 119, 255]
        ]
    )

    # THE DOTS (The backup so it's never "blank")
    scatterplot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nyc_df,
        get_position=["longitude", "latitude"],
        get_radius=50,
        get_fill_color=[255, 159, 229, 100], # Your pink
    )

    view_state = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=10,
        pitch=40
    )

    st.pydeck_chart(pdk.Deck(
        layers=[scatterplot_layer, heatmap_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/navigation-night-v1",
    ))

    

# -----------------------------
# PREDICTION
# -----------------------------
with col2:
    st.subheader("💰 Valuation Engine")
    st.write(
        "Random Forest model trained on 40,000+ NYC listings with "
        "semantic luxury scoring."
    )

    if st.button("RUN AI VALUATION 🚀", use_container_width=True):
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        input_df["latitude"] = lat
        input_df["longitude"] = lon
        input_df["minimum_nights"] = nights

        if "luxury_score" in input_df.columns:
            input_df["luxury_score"] = lux

        prediction = model.predict(input_df)

        st.metric(
            label="Estimated Nightly Rate",
            value=f"${prediction[0]:.2f}"
        )

        st.success("✅ Prediction generated.")
        st.snow()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.info(
    "📊 **Stack:** Scikit-learn, PyDeck Geospatial Visualization, "
    "Streamlit Deployment, NLP-derived Luxury Signals."
)




