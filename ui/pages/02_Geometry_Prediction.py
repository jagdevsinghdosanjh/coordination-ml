import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Geometry Prediction",
    page_icon="🔷",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict-geometry/"

st.title("🔷 Coordination Geometry Prediction")
st.write("Predict the geometry of a coordination complex using the Coordination‑ML API.")

st.divider()

# -----------------------------
# User Input Section
# -----------------------------
st.header("Input Parameters")

metal_symbol = st.text_input("Metal Symbol (e.g., Fe, Co, Ni, Pt)", "")
ligand_name = st.text_input("Ligand Name (exact match from ligands_raw.csv)", "")
coordination_number = st.number_input("Coordination Number", min_value=1, max_value=12, value=6)

submitted = st.button("Predict Geometry")

# -----------------------------
# Prediction Section
# -----------------------------
if submitted:
    payload = {
        "metal_symbol": metal_symbol.strip(),
        "ligand_name": ligand_name.strip(),
        "coordination_number": int(coordination_number)
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            st.error(f"API Error: {response.json().get('detail')}")
        else:
            result = response.json()

            st.success("Prediction Successful")

            st.subheader("Predicted Geometry")
            st.write(f"### {result['predicted_geometry']}")

            st.subheader("Confidence Score")
            st.progress(result["confidence"])
            st.write(f"**Confidence:** {result['confidence']:.4f}")

            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame(
                list(result["probabilities"].items()),
                columns=["Geometry", "Probability"]
            )
            st.dataframe(prob_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.divider()
st.caption("Powered by Coordination‑ML · Machine Learning for Inorganic Chemistry")

# import streamlit as st
# import pandas as pd
# from src.ml.inference import predict_geometry

# st.set_page_config(
#     page_title="Geometry Prediction",
#     page_icon="🔷",
#     layout="centered"
# )

# st.title("🔷 Coordination Geometry Prediction")
# st.write("Predict the geometry of a coordination complex using your trained ML model.")

# st.divider()

# # -----------------------------
# # User Input Section
# # -----------------------------
# st.header("Input Parameters")

# metal_symbol = st.text_input("Metal Symbol (e.g., Fe, Co, Ni, Pt)", "")
# ligand_name = st.text_input("Ligand Name (exact match from ligands_raw.csv)", "")
# coordination_number = st.number_input("Coordination Number", min_value=1, max_value=12, value=6)

# submitted = st.button("Predict Geometry")

# # -----------------------------
# # Prediction Section
# # -----------------------------
# if submitted:
#     try:
#         result = predict_geometry(
#             metal_symbol=metal_symbol.strip(),
#             ligand_name=ligand_name.strip(),
#             coordination_number=int(coordination_number)
#         )

#         st.success("Prediction Successful")

#         st.subheader("Predicted Geometry")
#         st.write(f"### {result['predicted_geometry']}")

#         st.subheader("Confidence Score")
#         st.progress(result["confidence"])
#         st.write(f"**Confidence:** {result['confidence']:.4f}")

#         st.subheader("Class Probabilities")
#         prob_df = pd.DataFrame(
#             list(result["probabilities"].items()),
#             columns=["Geometry", "Probability"]
#         )
#         st.dataframe(prob_df, use_container_width=True)

#     except Exception as e:
#         st.error(f"Error: {str(e)}")

# st.divider()

# st.caption("Powered by Coordination‑ML · Machine Learning for Inorganic Chemistry")
