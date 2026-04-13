import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('pro_prediction_model.pkl')

st.title('⚽ Betting Quant Engine')

h_shots = st.number_input("Home Shots on Target")
a_shots = st.number_input("Away Shots on Target")
h_corn = st.number_input("Home Corners")
a_corn = st.number_input("Away Corners")

if st.button("RUN PREDICTION"):
    features = pd.DataFrame([[h_shots - a_shots, h_corn - a_corn, 1]], 
                            columns=['Shots_Diff', 'Corners_Diff', 'Is_Home'])
    prob = model.predict_proba(features)[0][2]
    
    st.metric("Win Probability", f"{prob:.2%}")
    if prob > 0.45:
        st.success("✅ ACTION: Place the bet. The model sees an edge.")
    else:
        st.error("❌ ACTION: Stay away. No edge found.")
