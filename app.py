import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(page_title="Crop Advisory System", page_icon="🌾", layout="wide")

# Load the saved model
with open("model/crop_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
le = data["label_encoder"]
feature_names = data["features"]

st.title("🌾 Crop Advisory System for Indian Farmers")
st.write("Enter your soil and climate conditions to get a crop recommendation")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Soil Nutrients (kg/ha)")
    N = st.slider("Nitrogen (N)", 0, 140, 90)
    P = st.slider("Phosphorus (P)", 5, 145, 42)
    K = st.slider("Potassium (K)", 5, 205, 43)
    ph = st.slider("Soil pH", 3.5, 9.5, 6.5)

with col2:
    st.subheader("🌤️ Climate Conditions")
    temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0)
    humidity = st.slider("Humidity (%)", 14.0, 100.0, 80.0)
    rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 200.0)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🌱 Get Recommendation", "🔬 What-If Analysis", "📊 Data Insights"])

with tab1:
    if st.button("🌾 Get Crop Recommendation", use_container_width=True):
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                   columns=feature_names)
        input_scaled = scaler.transform(input_data)

        pred_encoded = model.predict(input_scaled)[0]
        pred_crop = le.inverse_transform([pred_encoded])[0]

        proba = model.predict_proba(input_scaled)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_crops = le.inverse_transform(top3_idx)
        top3_proba = proba[top3_idx]

        st.success(f"### 🏆 Recommended Crop: **{pred_crop.upper()}**")

        st.subheader("Top 3 Candidates")
        c1, c2, c3 = st.columns(3)
        for col, crop, prob in zip([c1, c2, c3], top3_crops, top3_proba):
            col.metric(label=crop.title(), value=f"{prob*100:.1f}%")

        st.markdown("---")
        st.subheader("📅 Crop Profile")

        seasons = {
            'rice': 'Kharif (June - November)', 'maize': 'Kharif (June - September)',
            'jute': 'Kharif (March - July)', 'cotton': 'Kharif (April - August)',
            'coconut': 'Year-round', 'papaya': 'Year-round', 'orange': 'Winter (Oct - Feb)',
            'apple': 'Winter (Oct - Feb)', 'muskmelon': 'Summer (Feb - May)',
            'watermelon': 'Summer (Feb - May)', 'grapes': 'Winter (Jan - Mar)',
            'mango': 'Summer (Mar - June)', 'banana': 'Year-round',
            'pomegranate': 'Year-round', 'lentil': 'Rabi (Oct - Mar)',
            'blackgram': 'Kharif/Rabi', 'mungbean': 'Kharif (Mar - June)',
            'mothbeans': 'Kharif (June - Sept)', 'pigeonpeas': 'Kharif (June - Dec)',
            'kidneybeans': 'Kharif (June - Oct)', 'chickpea': 'Rabi (Oct - Mar)',
            'coffee': 'Year-round'
        }

        water_req = {
            'rice': 'High', 'jute': 'High', 'coconut': 'High', 'banana': 'High', 'papaya': 'High',
            'maize': 'Medium', 'cotton': 'Medium', 'grapes': 'Medium', 'orange': 'Medium',
            'pomegranate': 'Medium', 'apple': 'Medium', 'mango': 'Medium', 'coffee': 'Medium',
            'chickpea': 'Low', 'lentil': 'Low', 'mothbeans': 'Low', 'mungbean': 'Low',
            'blackgram': 'Low', 'kidneybeans': 'Low', 'pigeonpeas': 'Low',
            'muskmelon': 'Low', 'watermelon': 'Medium'
        }

        ideal_states = {
            'rice': 'West Bengal, UP, Punjab', 'maize': 'MP, Karnataka, Bihar',
            'jute': 'West Bengal, Assam', 'cotton': 'Gujarat, Maharashtra, Telangana',
            'coconut': 'Kerala, Tamil Nadu', 'papaya': 'Andhra Pradesh, Karnataka',
            'orange': 'Maharashtra, MP', 'apple': 'Himachal Pradesh, J&K',
            'muskmelon': 'UP, Punjab, Haryana', 'watermelon': 'Karnataka, AP, UP',
            'grapes': 'Maharashtra, Karnataka', 'mango': 'UP, AP, Karnataka',
            'banana': 'Tamil Nadu, Maharashtra, Gujarat', 'pomegranate': 'Maharashtra, Karnataka',
            'lentil': 'MP, UP, West Bengal', 'blackgram': 'MP, Maharashtra, Rajasthan',
            'mungbean': 'Rajasthan, Maharashtra, Karnataka', 'mothbeans': 'Rajasthan, Gujarat',
            'pigeonpeas': 'Maharashtra, Karnataka, UP', 'kidneybeans': 'Karnataka, UP, Bihar',
            'chickpea': 'MP, Rajasthan, Maharashtra', 'coffee': 'Karnataka, Kerala, Tamil Nadu'
        }

        prices = {
            'rice': '₹1,800-2,200/quintal', 'maize': '₹1,500-1,900/quintal',
            'jute': '₹4,000-5,000/quintal', 'cotton': '₹6,000-7,500/quintal',
            'coconut': '₹2,500-3,500/quintal', 'papaya': '₹800-1,500/quintal',
            'orange': '₹2,000-3,000/quintal', 'apple': '₹4,000-8,000/quintal',
            'muskmelon': '₹1,000-2,000/quintal', 'watermelon': '₹800-1,500/quintal',
            'grapes': '₹3,000-6,000/quintal', 'mango': '₹2,000-5,000/quintal',
            'banana': '₹1,000-1,800/quintal', 'pomegranate': '₹5,000-9,000/quintal',
            'lentil': '₹5,500-6,500/quintal', 'blackgram': '₹6,000-7,000/quintal',
            'mungbean': '₹6,500-7,500/quintal', 'mothbeans': '₹5,000-6,000/quintal',
            'pigeonpeas': '₹6,000-7,000/quintal', 'kidneybeans': '₹7,000-9,000/quintal',
            'chickpea': '₹4,500-5,500/quintal', 'coffee': '₹12,000-18,000/quintal'
        }

        pc1, pc2 = st.columns(2)
        with pc1:
            st.write(f"**🗓️ Growing Season:** {seasons.get(pred_crop, 'N/A')}")
            st.write(f"**💧 Water Requirement:** {water_req.get(pred_crop, 'N/A')}")
        with pc2:
            st.write(f"**📍 Ideal States in India:** {ideal_states.get(pred_crop, 'N/A')}")
            st.write(f"**💰 Avg. Market Price:** {prices.get(pred_crop, 'N/A')}")

        st.markdown("---")
        st.subheader("📈 Confidence Gap Analysis")

        gap = (top3_proba[0] - top3_proba[1]) * 100
        if gap < 10:
            st.warning(f"⚠️ Borderline case! The gap between **{top3_crops[0].title()}** "
                       f"({top3_proba[0]*100:.1f}%) and **{top3_crops[1].title()}** "
                       f"({top3_proba[1]*100:.1f}%) is only **{gap:.1f}%**. "
                       f"Both crops are viable for these conditions.")
        else:
            st.info(f"✅ Confident recommendation. **{top3_crops[0].title()}** leads by "
                    f"**{gap:.1f}%** over the next best option, **{top3_crops[1].title()}**.")

        st.markdown("---")
        st.subheader("📝 Soil Report Summary")

        n_level = "high" if N > 80 else "low" if N < 40 else "moderate"
        p_level = "high" if P > 60 else "low" if P < 20 else "moderate"
        k_level = "high" if K > 60 else "low" if K < 20 else "moderate"
        ph_desc = "acidic" if ph < 5.5 else "alkaline" if ph > 7.5 else "neutral"
        rain_desc = "high" if rainfall > 200 else "low" if rainfall < 80 else "adequate"

        summary = (
            f"Your soil has **{n_level} Nitrogen ({N})**, **{p_level} Phosphorus ({P})**, "
            f"and **{k_level} Potassium ({K})**. Soil pH of **{ph}** is **{ph_desc}**. "
            f"Rainfall of **{rainfall}mm** is **{rain_desc}**, with a temperature of **{temperature}°C** "
            f"and humidity of **{humidity}%**. Based on these conditions, "
            f"**{pred_crop.title()}** is recommended with **{top3_proba[0]*100:.1f}% confidence**."
        )
        st.write(summary)

        st.markdown("---")
        st.subheader("🩺 Soil Health Summary")

        issues = []
        if ph < 5.5:
            issues.append(f"⚠️ Soil is **acidic** (pH={ph}). Most crops prefer pH 6-7.")
        elif ph > 7.5:
            issues.append(f"⚠️ Soil is **alkaline** (pH={ph}). Most crops prefer pH 6-7.")
        else:
            issues.append(f"✅ Soil pH ({ph}) is in the ideal range.")

        if N < 40:
            issues.append(f"⚠️ **Low Nitrogen** ({N}). May limit leafy growth.")
        if P < 20:
            issues.append(f"⚠️ **Low Phosphorus** ({P}). May affect root/flower development.")
        if K < 20:
            issues.append(f"⚠️ **Low Potassium** ({K}). May reduce disease resistance.")

        for issue in issues:
            st.write(issue)

with tab2:
    st.subheader("🔬 Interactive What-If Analysis")
    st.write("Adjust sliders to see live predictions update instantly")

    wcol1, wcol2 = st.columns(2)

    with wcol1:
        wN = st.slider("Nitrogen (N) ", 0, 140, N, key="wN")
        wP = st.slider("Phosphorus (P) ", 5, 145, P, key="wP")
        wK = st.slider("Potassium (K) ", 5, 205, K, key="wK")
        wtemp = st.slider("Temperature (°C) ", 8.0, 44.0, temperature, key="wtemp")
        whum = st.slider("Humidity (%) ", 14.0, 100.0, humidity, key="whum")
        wph = st.slider("Soil pH ", 3.5, 9.5, ph, key="wph")
        wrain = st.slider("Rainfall (mm) ", 20.0, 300.0, rainfall, key="wrain")

    with wcol2:
        w_input = pd.DataFrame([[wN, wP, wK, wtemp, whum, wph, wrain]], columns=feature_names)
        w_scaled = scaler.transform(w_input)

        w_pred = model.predict(w_scaled)[0]
        w_crop = le.inverse_transform([w_pred])[0]
        w_proba = model.predict_proba(w_scaled)[0]
        w_top3_idx = np.argsort(w_proba)[::-1][:3]
        w_top3_crops = le.inverse_transform(w_top3_idx)
        w_top3_proba = w_proba[w_top3_idx]

        st.markdown("#### Live Prediction")
        st.metric("Predicted Crop", w_crop.title())
        st.metric("Model Confidence", f"{w_top3_proba[0]*100:.1f}%")

        st.markdown("#### Top 3 Probability Distribution")
        for crop, prob in zip(w_top3_crops, w_top3_proba):
            st.write(f"**{crop.title()}**: {prob*100:.1f}%")
            st.progress(float(prob))

with tab3:
    st.subheader("📊 Dataset Insights")

    st.write("**Crop Distribution** — 22 crops, 100 samples each (balanced dataset)")
    st.write("**Correlation Heatmap** — Phosphorus and Potassium show the strongest correlation (0.74)")
    st.write("**Feature Importance** — Rainfall and Humidity are the top 2 predictors (~45% combined)")

    st.markdown("---")
    st.subheader("🤖 Model Performance")

    model_results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'SVM', 'KNN'],
        'Accuracy (%)': [97.27, 99.55, 98.41, 97.95]
    })
    st.dataframe(model_results, use_container_width=True, hide_index=True)
    st.success("🏆 Random Forest selected as the best model (99.55% accuracy)")