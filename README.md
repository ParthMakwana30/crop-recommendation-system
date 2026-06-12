# 🌾 Crop Recommendation System for Indian Farmers

An end-to-end machine learning project that recommends the optimal crop to grow based on soil nutrients (N, P, K), pH, and climate conditions (temperature, humidity, rainfall) — built with a focus on real-world relevance for Indian agriculture.

## 🔗 Live App

[https://your-app-name.streamlit.app](#) *(update after deployment)*

## 📌 Problem Statement

India has over 140 million farming households, most of whom make crop decisions based on tradition rather than soil and climate data. This project explores whether a simple machine learning model, trained on soil and environmental parameters, can recommend a more suitable crop — potentially improving yield and income.

## 📊 Dataset

[Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) — 2200 samples, 22 crops, 7 features:

* Nitrogen (N), Phosphorus (P), Potassium (K) — kg/ha
* Temperature (°C), Humidity (%), Soil pH, Rainfall (mm)

## 🔬 Approach

1. **Exploratory Data Analysis**

   * Crop distribution (balanced — 100 samples/crop)
   * Correlation heatmap (P-K correlation = 0.74)
   * Boxplots of N/P/K by crop — legumes show low N, high P/K (consistent with agronomy: legumes fix their own nitrogen)
   * Rainfall vs Temperature scatter — crops form distinct climate clusters
2. **Model Training \& Comparison**

|Model|Accuracy|
|-|-|
|Logistic Regression|97.27%|
|**Random Forest**|**99.55%** ✅|
|SVM|98.41%|
|KNN|97.95%|

3. **Feature Importance** (Random Forest)

   * Rainfall (23%) and Humidity (22.4%) are the top predictors — climate matters more than soil nutrients combined (42%)
   * pH ranked lowest (5%), despite being commonly emphasized in farming discussions
4. **Confusion Matrix**

   * Only 2 misclassifications out of 440 test samples — both involving Rice/Jute, which share similar high-rainfall, high-humidity profiles
5. **What-If Analysis**

   * Varying rainfall (20mm → 300mm) at fixed conditions shows a clear progression: Muskmelon → Watermelon → Jute → Rice, with the Jute→Rice threshold around 190mm — directly explaining the Rice/Jute confusion above

## 🖥️ Web App Features (Streamlit)

* **Crop Recommendation** — 7-parameter input, top-3 crops with confidence scores
* **Crop Profile** — growing season, water requirement, ideal Indian states, market price range
* **Confidence Gap Analysis** — flags borderline cases where two crops are nearly equally suitable
* **Soil Report Summary** — auto-generated plain-language summary of soil health
* **Interactive What-If Sandbox** — live sliders showing how predictions change in real time
* **Soil Health Check** — flags acidic/alkaline pH and low nutrient levels

## 🛠️ Tech Stack

* **Python** — Pandas, NumPy
* **ML** — Scikit-learn (Logistic Regression, Random Forest, SVM, KNN)
* **Visualization** — Matplotlib, Seaborn
* **Deployment** — Streamlit, Pickle

## 🚀 Run Locally

```bash
git clone https://github.com/ParthMakwana30/crop-recommendation-system.git
cd crop-recommendation-system
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
crop-recommendation-system/
├── DATA/
│   └── Crop\_recommendation.csv
├── model/
│   └── crop\_model.pkl
├── CROP\_EDA\_MODEL.ipynb     # Full EDA + model training notebook
├── app.py                    # Streamlit web app
├── requirements.txt
└── README.md
```

## 🔮 Future Improvements

* Replace Random Forest with XGBoost/LightGBM for potential accuracy gains
* Add cross-validation for more robust performance estimates
* Integrate real-time soil sensor / government agricultural API data
* Add multilingual support (Hindi, Marathi) for accessibility to actual farmers
* Expand to more crops and regional varieties using ICAR datasets

## 👤 Author

Built as part of a data science portfolio focused on applying ML to real Indian agricultural challenges.

