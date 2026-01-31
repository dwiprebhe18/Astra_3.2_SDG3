import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Poshan Tracker | RAG Analytics",
    layout="wide",
    page_icon="📊"
)

# --------------------------------------------------
# GLOBAL THEME & CSS
# --------------------------------------------------
st.markdown("""
<style>
.main { background-color: #f9fafc; }

.header-box {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    color: white;
    margin-bottom: 25px;
}

.metric-box {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    text-align: center;
}

section[data-testid="stSidebar"] {
    background-color: #f1f4f8;
}

h1, h2, h3 {
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------
@st.cache_data
def load_poshan_data():
    try:
        return pd.read_csv("mapped_poshan_tracker_updated (1).csv")
    except:
        return pd.DataFrame()

@st.cache_data
def load_synthetic_data():
    try:
        df = pd.read_csv("child_health_data.csv")
        # Map Malnutrition_Status to RAG_Status
        rag_map = {
            'Healthy': 'Green',
            'Underweight': 'Amber',
            'Wasted': 'Amber',
            'Severely Wasted': 'Red',
            'Stunted': 'Red'
        }
        df['RAG_Status'] = df['Malnutrition_Status'].map(rag_map).fillna('Green')
        return df
    except:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        return joblib.load("malnutrition_model.pkl"), joblib.load("encoders.pkl")
    except:
        return None, None

df_poshan = load_poshan_data()
df_syn = load_synthetic_data()
model, encoders = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("📌 Poshan Operations")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 National Insights", "🧒 Community RAG Status", "🤖 AI Risk Prediction"]
)

# --------------------------------------------------
# PAGE 1: NATIONAL DASHBOARD
# --------------------------------------------------
if page == "📊 National Insights":

    st.markdown("""
    <div class="header-box">
        <h1>Poshan Abhiyaan – National Dashboard</h1>
        <p>Real-time insights from Anganwadi & beneficiary data</p>
    </div>
    """, unsafe_allow_html=True)

    if df_poshan.empty:
        st.warning("Dataset not found.")
    else:
        states = ["All"] + sorted(df_poshan["state_name"].unique())
        selected_state = st.sidebar.selectbox("Filter by State", states)

        df_view = df_poshan if selected_state == "All" else df_poshan[df_poshan["state_name"] == selected_state]

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏠 AWCs", f"{df_view['awc'].sum():,}")
        k2.metric("👨‍👩‍👧 Beneficiaries", f"{df_view['elgb_benef'].sum():,}")
        k3.metric("🚽 Toilets", f"{df_view['awc_infra_fun_toilets'].sum():,}")
        k4.metric("🚰 Drinking Water", f"{df_view['awc_infra_dws'].sum():,}")

        st.markdown("---")

        # Beneficiary Pie
        ben_cols = {
            'children_0_6_months': '0–6 Months',
            'children_6months_3years': '6M–3Y',
            'children_3_6_years': '3–6Y',
            'pregnant_women': 'Pregnant',
            'lact_mothers': 'Lactating',
            'adolescent_girls': 'Adolescent Girls'
        }

        ben_df = df_view[list(ben_cols)].sum().reset_index()
        ben_df.columns = ["Category", "Count"]
        ben_df["Label"] = ben_df["Category"].map(ben_cols)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("👥 Beneficiary Distribution")
            # Interactive Pie Chart with Pull
            pull_values = [0.0] * len(ben_df)
            selection = st.session_state.get("pie_selection", None)
            
            if selection and selection.get("selection") and selection["selection"]["points"]:
                selected_i = selection["selection"]["points"][0]["point_number"]
                pull_values[selected_i] = 0.2
            
            fig_pie = px.pie(
                ben_df,
                values="Count",
                names="Label",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_traces(
                textinfo="percent+label",
                pull=pull_values
            )
            fig_pie.update_layout(
                clickmode='event+select'
            )
            st.plotly_chart(fig_pie, on_select="rerun", key="pie_selection", width='stretch')

        with c2:
            st.subheader("🏗️ Infrastructure Coverage")
            infra_df = pd.DataFrame({
                "Type": ["Total AWCs", "Water", "Toilets", "Own Building"],
                "Count": [
                    df_view["awc"].sum(),
                    df_view["awc_infra_dws"].sum(),
                    df_view["awc_infra_fun_toilets"].sum(),
                    df_view["awc_infra_own_buil"].sum()
                ]
            })
            fig_bar = px.bar(infra_df, x="Type", y="Count", text_auto=True,
                             color_discrete_sequence=["#1f77b4"])
            st.plotly_chart(fig_bar, width='stretch')

        st.markdown("---")

        st.subheader("🔥 Operational Correlation Heatmap")
        corr_cols = [
            'awc', 'awc_infra_fun_toilets',
            'children_0_6_months', 'children_3_6_years_school'
        ]
        fig_heat = px.imshow(
            df_view[corr_cols].corr(),
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_heat, width='stretch')

# --------------------------------------------------
# PAGE 2: COMMUNITY RAG
# --------------------------------------------------
elif page == "🧒 Community RAG Status":

    st.markdown("## 📍 Community Malnutrition Overview")

    if df_syn.empty:
        st.warning("Synthetic data not found.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Children", len(df_syn))
        c2.metric("🔴 Red", (df_syn["RAG_Status"] == "Red").sum())
        c3.metric("🟠 Amber", (df_syn["RAG_Status"] == "Amber").sum())
        c4.metric("🟢 Green", (df_syn["RAG_Status"] == "Green").sum())

        rag_df = df_syn["RAG_Status"].value_counts().reset_index()
        rag_df.columns = ['Status', 'Count']
        
        fig = px.pie(
            rag_df,
            values="Count",
            names="Status",
            hole=0.4,
            color="Status",
            color_discrete_map={
                "Red": "#ff4b4b",
                "Amber": "#ffa500",
                "Green": "#2ecc71"
            }
        )
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(fig, width='stretch')

# --------------------------------------------------
# PAGE 3: AI PREDICTION
# --------------------------------------------------
else:

    st.markdown("## 🤖 AI-Based Growth Risk Prediction")

    if model and encoders:
        with st.form("predict"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age (months)", 1, 60, 24)
                gender = st.selectbox("Gender", encoders['gender'].classes_)
                region = st.selectbox("Region", encoders['region'].classes_)
            with col2:
                height = st.number_input("Height (cm)", 40.0, 120.0, 80.0)
                weight = st.number_input("Weight (kg)", 2.0, 30.0, 10.0)
                socio = st.selectbox("Socio-Economic Status", encoders['socio'].classes_)
                sanitation = st.selectbox("Sanitation Access", encoders['sanitation'].classes_)

            submit = st.form_submit_button("Predict Risk Status")

            if submit:
                with st.spinner("Analyzing child health data..."):
                    X = pd.DataFrame([{
                        "Age_Months": age,
                        "Gender": encoders['gender'].transform([gender])[0],
                        "Height_cm": height,
                        "Weight_kg": weight,
                        "Region": encoders['region'].transform([region])[0],
                        "Socio_Economic_Status": encoders['socio'].transform([socio])[0],
                        "Sanitation_Access": encoders['sanitation'].transform([sanitation])[0]
                    }])

                    pred_encoded = model.predict(X)[0]
                    pred_mal = pred_encoded  # model.predict returns the class label directly
                    
                    rag_map = {
                        'Healthy': 'Green',
                        'Underweight': 'Amber',
                        'Wasted': 'Amber',
                        'Severely Wasted': 'Red',
                        'Stunted': 'Red'
                    }
                    pred_label = rag_map.get(pred_mal, 'Green')
                    prob = model.predict_proba(X).max()

                # Display results with animations
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                if pred_label == 'Red':
                    st.error(f"🔴 **High Risk: {pred_mal}** (Confidence: {prob:.2f})")
                    st.warning("⚠️ **Immediate Action Required:** Contact local health worker for nutritional assessment.")
                    st.markdown("**Recommendations:**")
                    st.markdown("- Schedule immediate medical checkup")
                    st.markdown("- Provide nutritional supplements")
                    st.markdown("- Monitor weight and height weekly")
                elif pred_label == 'Amber':
                    st.warning(f"🟠 **Moderate Risk: {pred_mal}** (Confidence: {prob:.2f})")
                    st.info("📋 **Monitoring Required:** Regular health checkups recommended.")
                    st.markdown("**Recommendations:**")
                    st.markdown("- Bi-weekly health monitoring")
                    st.markdown("- Nutritional counseling")
                    st.markdown("- Improved sanitation if applicable")
                else:
                    st.success(f"🟢 **Normal: {pred_mal}** (Confidence: {prob:.2f})")
                    st.info("✅ **Good Health Status:** Continue regular monitoring.")
                    st.markdown("**Recommendations:**")
                    st.markdown("- Maintain balanced nutrition")
                    st.markdown("- Regular growth monitoring")
                    st.markdown("- Age-appropriate vaccinations")

                # Visual confidence meter
                st.markdown("**AI Confidence Level:**")
                st.progress(prob)
                st.caption(f"{prob:.1%} confidence in this assessment")
