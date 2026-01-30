import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(
    page_title="Child Malnutrition Analytics",
    page_icon="👶",
    layout="wide"
)

# Load Data and Model
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('child_health_data.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run data_generator.py.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load('malnutrition_model.pkl')
        encoders = joblib.load('encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        st.error("Model files not found. Please run model_trainer.py.")
        return None, None

df = load_data()
model, encoders = load_model()

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Predictive Analytics", "Reports & Alerts"])

# --- DASHBOARD OVERVIEW ---
if page == "Dashboard Overview":
    st.title("📊 Child Malnutrition Tracker - Dashboard")
    
    if not df.empty:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        total_children = len(df)
        malnutrition_cases = df[df['Malnutrition_Status'] != 'Healthy']
        malnutrition_rate = (len(malnutrition_cases) / total_children) * 100
        critical_cases = df[df['Malnutrition_Status'].isin(['Severely Wasted', 'Stunted'])]
        
        col1.metric("Total Children Tracked", f"{total_children:,}")
        col2.metric("Malnutrition Rate", f"{malnutrition_rate:.1f}%")
        col3.metric("Critical Cases", f"{len(critical_cases)}")
        col4.metric("Regions Covered", f"{df['Region'].nunique()}")
        
        st.markdown("---")
        
        # Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Malnutrition Status Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df, x='Malnutrition_Status', palette='viridis', ax=ax1)
            plt.xticks(rotation=45)
            st.pyplot(fig1)
            
        with col_right:
            st.subheader("Regional Breakdown")
            fig2, ax2 = plt.subplots()
            # Calculate malnutrition rate per region
            region_stats = df.groupby('Region')['Malnutrition_Status'].apply(lambda x: (x != 'Healthy').mean() * 100).reset_index()
            sns.barplot(data=region_stats, x='Region', y='Malnutrition_Status', palette='magma', ax=ax2)
            ax2.set_ylabel("Malnutrition Rate (%)")
            st.pyplot(fig2)
            
        st.subheader("Correlation Analysis: Socio-Economic Factors")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.heatmap(pd.crosstab(df['Socio_Economic_Status'], df['Malnutrition_Status'], normalize='index'), annot=True, cmap='YlGnBu', fmt='.2f', ax=ax3)
        st.pyplot(fig3)

# --- PREDICTIVE ANALYTICS ---
elif page == "Predictive Analytics":
    st.title("🤖 Predict Malnutrition Risk")
    st.markdown("Enter child health details to predict current risk status.")
    
    if model and encoders:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age (Months)", min_value=1, max_value=60, value=24)
                gender = st.selectbox("Gender", encoders['gender'].classes_)
                region = st.selectbox("Region", encoders['region'].classes_)
                
            with col2:
                height = st.number_input("Height (cm)", min_value=40.0, max_value=120.0, value=85.0)
                weight = st.number_input("Weight (kg)", min_value=2.0, max_value=30.0, value=12.0)
                socio = st.selectbox("Socio-Economic Status", encoders['socio'].classes_)
                sanitation = st.selectbox("Sanitation Access", encoders['sanitation'].classes_)
            
            submitted = st.form_submit_button("Predict Status")
            
            if submitted:
                # Prepare input
                input_data = pd.DataFrame({
                    'Age_Months': [age],
                    'Gender': [encoders['gender'].transform([gender])[0]],
                    'Height_cm': [height],
                    'Weight_kg': [weight],
                    'Region': [encoders['region'].transform([region])[0]],
                    'Socio_Economic_Status': [encoders['socio'].transform([socio])[0]],
                    'Sanitation_Access': [encoders['sanitation'].transform([sanitation])[0]]
                })
                
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data).max()
                
                st.markdown("### Prediction Result")
                if prediction == 'Healthy':
                    st.success(f"**predicted Status: {prediction}** (Confidence: {proba:.2f})")
                else:
                    st.error(f"**Predicted Status: {prediction}** (Confidence: {proba:.2f})")
                    st.warning("⚠️ Recommendation: Immediate nutritional assessment required. Contact local health worker.")

# --- REPORTS & ALERTS ---
elif page == "Reports & Alerts":
    st.title("📢 Alerts & Actionable Insights")
    
    if not df.empty:
        # Identify High Risk Areas
        high_risk_regions = df[df['Malnutrition_Status'] != 'Healthy'].groupby('Region').size().sort_values(ascending=False).head(3)
        
        st.subheader("🚨 Priority Areas Integration")
        for region, count in high_risk_regions.items():
            st.warning(f"**{region} Region**: {count} reported cases of malnutrition. Prioritize food aid distribution.")
            
        st.markdown("---")
        
        st.subheader("📋 Recent Critical Cases")
        critical_df = df[df['Malnutrition_Status'].isin(['Severely Wasted', 'Stunted'])].head(10)
        st.dataframe(critical_df[['Child_ID', 'Age_Months', 'Region', 'Malnutrition_Status', 'Weight_kg']])
        
        st.download_button(
            label="Download Full Report (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='malnutrition_report.csv',
            mime='text/csv',
        )

st.sidebar.markdown("---")
st.sidebar.info("Developed for SDG 3: Good Health and Well-being")
