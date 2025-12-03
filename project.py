import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ESG Revenue Predictor", layout="wide")

st.title("📊 Corporate Revenue Prediction Dashboard")
st.markdown("""
This app loads a pre-trained **Random Forest** model to predict a company's revenue based on its 
**Financials**, **ESG Scores**, and **Operational Metrics**.
""")

# --- 1. LOAD ASSETS (Model, Features, and Data for Dropdowns) ---
# Added show_spinner=False to handle UI outside, or set to True for automatic spinner
@st.cache_resource(show_spinner="Loading models and data...")
def load_assets():
    # --- Load the Pre-trained Model ---
    model = None
    feature_names = []
    
    # We define status messages to return, rather than displaying them immediately
    model_status = "ok"
    
    try:
        model = joblib.load('random_forest_revenue_model.pkl')
        feature_names = getattr(model, 'feature_names_in_', [])
    except FileNotFoundError:
        model = None
        feature_names = []
        model_status = "missing_model"

    # --- Load Data ---
    df_raw = None
    data_source_name = None

    # Try multiple known locations for the dataset
    csv_paths = [
        r'C:\Users\user\OneDrive\Desktop\FINAL PROJECT\company_esg_financial_dataset.csv',
        'company_esg_financial_dataset.csv'
    ]
    
    for path in csv_paths:
        try:
            df_raw = pd.read_csv(path)
            data_source_name = os.path.basename(path)
            break
        except FileNotFoundError:
            continue

    if df_raw is None:
        # Create empty fallback
        df_raw = pd.DataFrame(columns=['Industry', 'Region', 'Revenue'])
        data_status = "missing_data"
    else:
        data_status = f"loaded_from_{data_source_name}"

    return model, feature_names, df_raw, model_status, data_status

# Execute loading
# We now unpack 5 values instead of 3 to handle the status messages safely
model, model_columns, df_raw, model_status, data_status = load_assets()

# --- HANDLE UI MESSAGES HERE (Outside the cached function) ---

# Check Model Status
if model_status == "ok":
    st.toast("Model loaded successfully!", icon="✅")
elif model_status == "missing_model":
    st.error("❌ Could not find the model file 'random_forest_revenue_model.pkl'.")
    st.stop() # Stop execution here if model is critical

# Check Data Status
if "loaded_from" in data_status:
    # Extract filename from status string for display
    fname = data_status.replace("loaded_from_", "")
    st.toast(f"Data loaded from '{fname}'", icon="📦")
elif data_status == "missing_data":
    st.warning("Could not find 'company_esg_financial_dataset.csv' in known locations; using empty fallback.")

# --- 2. SIDEBAR: USER INPUTS ---
st.sidebar.header("📝 Input Company Parameters")

def user_input_features():
    # Financials
    market_cap = st.sidebar.number_input("Market Cap ($M)", min_value=10.0, value=5000.0)
    growth_rate = st.sidebar.slider("Growth Rate (%)", min_value=-20.0, max_value=50.0, value=3.5)
    
    # ESG Scores
    st.sidebar.subheader("ESG Scores")
    esg_env = st.sidebar.slider("Environmental Score", 0, 100, 70)
    esg_soc = st.sidebar.slider("Social Score", 0, 100, 60)
    esg_gov = st.sidebar.slider("Governance Score", 0, 100, 65)
    
    # Operations
    st.sidebar.subheader("Operational Metrics")
    energy = st.sidebar.number_input("Energy Consumption", value=50000.0)
    carbon = st.sidebar.number_input("Carbon Emissions", value=20000.0)
    water = st.sidebar.number_input("Water Usage", value=10000.0)
    
    # Categorical Selection
    # Safety check if dataframe is empty
    if not df_raw.empty and 'Industry' in df_raw.columns:
        industry_options = df_raw['Industry'].unique()
    else:
        industry_options = ['Technology', 'Retail', 'Finance']
        
    if not df_raw.empty and 'Region' in df_raw.columns:
        region_options = df_raw['Region'].unique()
    else:
        region_options = ['North America', 'Asia', 'Europe']
    
    industry = st.sidebar.selectbox("Industry", industry_options)
    region = st.sidebar.selectbox("Region", region_options)
    
    # Store in dictionary
    data = {
        'MarketCap': market_cap,
        'GrowthRate': growth_rate,
        'ESG_Environmental': esg_env,
        'ESG_Social': esg_soc,
        'ESG_Governance': esg_gov,
        'EnergyConsumption': energy,
        'CarbonEmissions': carbon,
        'WaterUsage': water,
        'Industry': industry,
        'Region': region,
        'ESG_Balance_Std': np.std([esg_env, esg_soc, esg_gov])
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display User Input
st.subheader("Your Input Configuration")
st.write(input_df)

# --- 3. PREDICTION LOGIC ---
if st.button("🚀 Predict Revenue"):
    # Align columns with model
    input_encoded = pd.get_dummies(input_df)
    
    # Ensure all columns expected by the model exist, fill missing with 0
    # Note: reindex is safer than assuming columns match perfectly
    if len(model_columns) > 0:
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_encoded)
    
    st.success(f"💰 Predicted Annual Revenue: **${prediction[0]:,.2f}**")
    
    # --- 4. EXPLAINABILITY ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Comparison to Industry Average")
        if not df_raw.empty:
            # Safe filter
            industry_data = df_raw[df_raw['Industry'] == input_df['Industry'][0]]
            if not industry_data.empty:
                industry_avg = industry_data['Revenue'].mean()
                
                fig, ax = plt.subplots()
                bars = ['Predicted', 'Industry Avg']
                values = [prediction[0], industry_avg]
                colors = ['#4CAF50', '#FFC107']
                ax.bar(bars, values, color=colors)
                ax.set_ylabel("Revenue")
                st.pyplot(fig)
            else:
                st.info("No historical data for this industry.")
        else:
            st.warning("Cannot compare: Source data not loaded.")
        
    with col2:
        st.subheader("Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Get top 5 features
            indices = np.argsort(importances)[-5:] 
            
            fig2, ax2 = plt.subplots()
            ax2.barh(range(len(indices)), importances[indices], align='center', color='#2196F3')
            ax2.set_yticks(range(len(indices)))
            
            # Safe check for feature names length
            if len(model_columns) > 0:
                ax2.set_yticklabels([model_columns[i] for i in indices])
            
            ax2.set_xlabel('Relative Importance')
            st.pyplot(fig2)