import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Ligand Strength Prediction",
    page_icon="🔶",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict-ligand-strength/"

st.title("🔶 Ligand Strength Prediction")
st.write("Predict ligand donor strength using ML and atomic descriptors.")

st.divider()

# -----------------------------
# User Input Section
# -----------------------------
st.header("Input Parameters")

ligand_name = st.text_input("Ligand Name (exact match from ligands_raw.csv)", "")

submitted = st.button("Predict Strength")

# -----------------------------
# Prediction Section
# -----------------------------
if submitted:
    payload = {
        "ligand_name": ligand_name.strip()
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            st.error(f"API Error: {response.json().get('detail')}")
        else:
            result = response.json()

            st.success("Prediction Successful")

            st.subheader("Ligand Strength Score")
            st.write(f"### {result['strength_score']:.4f}")

            st.subheader("Features Used")
            features_df = pd.DataFrame(
                list(result["features_used"].items()),
                columns=["Feature", "Value"]
            )
            st.dataframe(features_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.divider()
st.caption("Powered by Coordination‑ML · Machine Learning for Inorganic Chemistry")
