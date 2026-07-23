import streamlit as st

st.set_page_config(
    page_title="Coordination‑ML",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 Coordination‑ML")
st.write("Machine Learning tools for coordination chemistry.")

st.divider()

st.header("Available Tools")

st.markdown("""
### 🔷 Geometry Prediction  
Predict the geometry (octahedral, tetrahedral, square planar) of a coordination complex using ML.

Go to: **Pages → 02_Geometry_Prediction**
""")

st.markdown("""
### 🔶 Ligand Strength Prediction *(coming soon)*  
Estimate ligand strength using donor atom descriptors and Δ₀ values.
""")

st.markdown("""
### 🔵 Stability Prediction *(coming soon)*  
Predict log(Kf) stability constants using ML regression.
""")

st.divider()

st.caption("Powered by Coordination‑ML · Built by Jagdev Singh Dosanjh")
