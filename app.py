import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Poshan Tracker - RAG Analytics",

    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Processed Synthetic Data (for Individual Risk Prediction)
@st.cache_data
def load_synthetic_data():
    try:
        df = pd.read_csv('processed_health_data.csv')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# 2. Load Real Poshan Dataset (for Aggregated Insights)
@st.cache_data
def load_poshan_data():
    try:
        # Using the filename provided by user
        df = pd.read_csv('mapped_poshan_tracker_updated (1).csv')
        # Basic cleanup if needed
        return df
    except FileNotFoundError:
        st.error("Poshan dataset file not found.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load('malnutrition_model.pkl')
        encoders = joblib.load('encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        return None, None

# Load datasets
df_synthetic = load_synthetic_data()
df_poshan = load_poshan_data()
model, encoders = load_model()

# Sidebar
st.sidebar.title("Poshan Operations")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["Poshan Insights (Real Data)", "Community RAG Status", "Risk Prediction (AI)"])

# --- PAGE 1: POSHAN INSIGHTS (REAL DATA) ---
if page == "Poshan Insights (Real Data)":
    st.title("Poshan Abhiyaan - National Dashboard")
   
    if df_poshan.empty:
        st.warning("Poshan dataset not loaded. Please check the file path.")
    else:
        # Filters
        states = ['All'] + sorted(df_poshan['state_name'].unique().tolist())
        selected_state = st.sidebar.selectbox("Filter by State", states)
       
        if selected_state != 'All':
            df_view = df_poshan[df_poshan['state_name'] == selected_state]
        else:
            df_view = df_poshan

        # KPI Metrics
        total_centers = df_view['awc'].sum()
        total_beneficiaries = df_view['elgb_benef'].sum()
        infra_toilet = df_view['awc_infra_fun_toilets'].sum()
        infra_water = df_view['awc_infra_dws'].sum()
       
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anganwadi Centers (AWC)", f"{total_centers:,}")
        c2.metric("Total Beneficiaries", f"{total_beneficiaries:,}")
        c3.metric("Functional Toilets", f"{infra_toilet:,}")
        c4.metric("Drinking Water Access", f"{infra_water:,}")
       
        st.markdown("---")

        # 1. PIE CHART: Beneficiary Distribution (Interactive with Plotly)
        st.subheader("Beneficiary Composition")
       
        # Columns representing different beneficiary types
        ben_cols = {
            'children_0_6_months': 'Children (0-6M)',
            'children_6months_3years': 'Children (6M-3Y)',
            'children_3_6_years': 'Children (3-6Y)',
            'pregnant_women': 'Pregnant Women',
            'lact_mothers': 'Lactating Mothers',
            'adolescent_girls': 'Adolescent Girls'
        }
       
        # Summing up for the current view
        ben_data = df_view[list(ben_cols.keys())].sum().reset_index()
        ben_data.columns = ['Category_Col', 'Count']
        ben_data['Category_Label'] = ben_data['Category_Col'].map(ben_cols)
       
        # layout symmetry: Equal width columns to reduce asymmetric whitespace
        c_pie, c_bar = st.columns(2)
        with c_pie:
            # Using Plotly Pie Chart (Standard Pie, not Donut)
            # Create base figure first to get indexing right
            fig_pie = px.pie(
                ben_data,
                values='Count',
                names='Category_Label',
                title="Beneficiary Breakdown",
                color_discrete_sequence=px.colors.qualitative.Bold, # Stronger colors
                # hole=0.0 # Removed hole to make it a full Pie Chart
            )
           
            # --- INTERACTION LOGIC ---
            pull_values = [0.0] * len(ben_data)
            selection = st.session_state.get("pie_selection", None)
           
            if selection and selection.get("selection") and selection["selection"]["points"]:
                # Get the point index of the click
                selected_i = selection["selection"]["points"][0]["point_number"]
                pull_values[selected_i] = 0.2 # Pop out!
           
            fig_pie.update_traces(
                textposition='auto',  # 'auto' places inside if big enough, outside if small
                textinfo='percent+label',
                pull=pull_values,
                hoverinfo='label+percent+value',
                marker=dict(line=dict(color='#000000', width=1))
            )
            fig_pie.update_layout(
                showlegend=True, # Show legend to help if labels are cluttered
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), # Legend at bottom
                margin=dict(t=40, b=50, l=40, r=40), # Increased margins to prevent clipping
                clickmode='event+select' # Enable selection events
            )
           
            # Render and capture selection
            # key="pie_selection" automagically stores the result in st.session_state["pie_selection"]
            st.plotly_chart(fig_pie, width="stretch", on_select="rerun", key="pie_selection")
           
        # 2. BAR CHART: Infrastructure vs Centers (Interactive with Plotly)
        with c_bar:
            st.subheader(f"AWC Infrastructure Coverage ({selected_state})")
           
            infra_data = pd.DataFrame({
                'Metric': ['Total AWCs', 'Drinking Water', 'Functional Toilets', 'Own Building'],
                'Count': [
                    df_view['awc'].sum(),
                    df_view['awc_infra_dws'].sum(),
                    df_view['awc_infra_fun_toilets'].sum(),
                    df_view['awc_infra_own_buil'].sum()
                ]
            })
           
            # Using Plotly Bar Chart with SOLID COLORS
            fig_bar = px.bar(
                infra_data,
                x='Metric',
                y='Count',
                # Removing 'color' argument to avoid gradient. Setting specific marker color.
                title="Infrastructure Availability",
            )
            # Update traces for solid, high-visibility color (e.g., RoyalBlue)
            fig_bar.update_traces(marker_color='rgb(26, 118, 255)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=1.0)
           
            fig_bar.update_layout(
                clickmode='event+select',
                showlegend=False,
                margin=dict(t=40, b=0, l=20, r=20)
            )
            st.plotly_chart(fig_bar, width="stretch")

        st.markdown("---")
       
        # 3. HEATMAP: Correlation Analysis (Interactive with Plotly)
        st.subheader("Operational Indicators Heatmap")
       
        # Select numeric columns for correlation
        corr_cols = [
            'awc', 'awc_infra_fun_toilets', 'children_0_6_months',
            'children_3_6_years_school', 'hcm_atleast_21days', 'thm_atleast_21days'
        ]
       
        # Compute correlation matrix
        corr_matrix = df_view[corr_cols].corr()
       
        # Using Plotly Heatmap
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_heat, width="stretch")
       
        st.info("Interactive Heatmap: Hover over cells to see exact correlation values.")

# --- PAGE 2: COMMUNITY RAG STATUS (SYNTHETIC SCENARIO) ---
elif page == "Community RAG Status":
    st.title("Local Community - Malnutrition Status")
    st.caption("Based on processed individual child records (Synthetic Source)")
   
    if not df_synthetic.empty:
        # Top Metrics
        total = len(df_synthetic)
        red_count = len(df_synthetic[df_synthetic['RAG_Status'] == 'Red'])
        amber_count = len(df_synthetic[df_synthetic['RAG_Status'] == 'Amber'])
        green_count = len(df_synthetic[df_synthetic['RAG_Status'] == 'Green'])
       
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tracked", total)
        c2.metric("Red (Severe)", red_count)
        c3.metric("Amber (Mod.)", amber_count)
        c4.metric("Green (Normal)", green_count)
       
        st.markdown("---")
       
        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("RAG Status Distribution")
            # Using Plotly Bar for RAG
            rag_counts = df_synthetic['RAG_Status'].value_counts().reset_index()
            rag_counts.columns = ['Status', 'Count']
           
            color_map = {'Red': '#ff4b4b', 'Amber': '#ffa500', 'Green': '#09ab3b'}
           
            fig_rag = px.bar(
                rag_counts,
                x='Status',
                y='Count',
                color='Status',
                color_discrete_map=color_map,
                category_orders={"Status": ["Red", "Amber", "Green"]}
            )
            st.plotly_chart(fig_rag, width="stretch")
           
        with c_right:
            st.subheader("Priority List (Red Zone)")
            st.dataframe(df_synthetic[df_synthetic['RAG_Status'] == 'Red'][['Child_ID', 'Age_Months', 'Weight_kg', 'Region']].head(10))

# --- PAGE 3: RISK PREDICTION (ML) ---
elif page == "Risk Prediction (AI)":
    st.title("AI Growth Assessment")
    st.info("Predicts Malnutrition Risk using Random Forest Model (Trained on Synthetic Data)")
   
    if model and encoders:
        with st.form("assess_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (Months)", 1, 60, 24)
                gender = st.selectbox("Gender", encoders['gender'].classes_)
                region = st.selectbox("Region", encoders['region'].classes_)
            with c2:
                height = st.number_input("Height (cm)", 40.0, 120.0, 80.0)
                weight = st.number_input("Weight (kg)", 2.0, 30.0, 10.0)
                socio = st.selectbox("Socio-Economic", encoders['socio'].classes_)
                sanitation = st.selectbox("Sanitation Access", encoders['sanitation'].classes_)
               
            submit = st.form_submit_button("Predict Status")
           
            if submit:
                input_df = pd.DataFrame([{
                    'Age_Months': age,
                    'Gender': encoders['gender'].transform([gender])[0],
                    'Height_cm': height,
                    'Weight_kg': weight,
                    'Region': encoders['region'].transform([region])[0],
                    'Socio_Economic_Status': encoders['socio'].transform([socio])[0],
                    'Sanitation_Access': encoders['sanitation'].transform([sanitation])[0]
                }])
               
                pred_encoded = model.predict(input_df)[0]
                # Since model predicts encoded Malnutrition_Status, map to RAG
                malnutrition_labels = ['Healthy', 'Severely Wasted', 'Stunted', 'Underweight', 'Wasted']  # Assuming order
                if pred_encoded < len(malnutrition_labels):
                    pred_malnutrition = malnutrition_labels[pred_encoded]
                else:
                    pred_malnutrition = 'Healthy'
                
                rag_map = {
                    'Healthy': 'Green',
                    'Underweight': 'Amber',
                    'Wasted': 'Amber',
                    'Severely Wasted': 'Red',
                    'Stunted': 'Red'
                }
                pred_label = rag_map.get(pred_malnutrition, 'Green')
                probs = model.predict_proba(input_df).max()
               
                if pred_label == 'Red':
                    st.error(f"Status: {pred_label} (Confidence: {probs:.2f})")
                elif pred_label == 'Amber':
                    st.warning(f"Status: {pred_label} (Confidence: {probs:.2f})")
                else:
                    st.success(f"Status: {pred_label} (Confidence: {probs:.2f})")
