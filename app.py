import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    with open('lifestyle_model.pkl', 'rb') as f:
        lifestyle_model = pickle.load(f)
    with open('clinical_model.pkl', 'rb') as f:
        clinical_model = pickle.load(f)
    with open('scaler_life.pkl', 'rb') as f:
        scaler_life = pickle.load(f)
    with open('scaler_clin.pkl', 'rb') as f:
        scaler_clin = pickle.load(f)
    
    try:
        with open('feature_names.pkl', 'rb') as f:
            feature_info = pickle.load(f)
    except:
        feature_info = None
    
    return lifestyle_model, clinical_model, scaler_life, scaler_clin, feature_info

lifestyle_model, clinical_model, scaler_life, scaler_clin, feature_info = load_models()

# Title
st.title("🫀 Two-Stage Heart Disease Risk Assessment")
st.markdown("---")

# Show feature info
if feature_info:
    with st.expander("ℹ️ Model Information"):
        st.write("**Stage 1 Features:**", feature_info.get('lifestyle_features', []))
        st.write("**Stage 2 Features:**", feature_info.get('clinical_features', []))

# Sidebar
st.sidebar.header("Navigation")
stage = st.sidebar.radio("Select Stage", ["Stage 1: Lifestyle Screening", "Stage 2: Clinical Diagnosis"])

# ============================================================
# STAGE 1: LIFESTYLE SCREENING
# ============================================================
if stage == "Stage 1: Lifestyle Screening":
    st.header("📊 Stage 1: Lifestyle Risk Screening")
    st.info("This assessment evaluates heart disease risk based on your health profile.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.number_input("Age (years)", 18, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        
        st.subheader("Cholesterol Levels")
        total_cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
        ldl = st.number_input("LDL Cholesterol (mg/dL)", 50, 300, 130)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 50)
    
    with col2:
        st.subheader("Blood Pressure")
        systolic_bp = st.number_input("Systolic BP (mm Hg)", 80, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP (mm Hg)", 50, 130, 80)
        
        st.subheader("Risk Factors")
        smoking = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    
    if st.button("Calculate Lifestyle Risk", type="primary"):
        # Prepare input - FIX: Create DataFrame with feature names
        sex_encoded = 1 if sex == "Male" else 0
        smoking_encoded = 1 if smoking == "Smoker" else 0
        diabetes_encoded = 1 if diabetes == "Yes" else 0
        
        # Create DataFrame with proper feature names
        lifestyle_data = pd.DataFrame([[
            age,
            sex_encoded,
            total_cholesterol,
            ldl,
            hdl,
            systolic_bp,
            diastolic_bp,
            smoking_encoded,
            diabetes_encoded
        ]], columns=feature_info['lifestyle_features'])
        
        # Scale and predict
        lifestyle_scaled = scaler_life.transform(lifestyle_data)
        p_l = lifestyle_model.predict_proba(lifestyle_scaled)[0][1]
        
        # Store in session state
        st.session_state.p_l = p_l
        
        # Display result
        st.markdown("---")
        st.subheader("Lifestyle Risk Assessment Results")
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=p_l * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Lifestyle Risk Score (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        threshold = 0.6
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score", f"{p_l*100:.1f}%")
        with col2:
            st.metric("Threshold", "60%")
        with col3:
            if p_l < threshold:
                st.metric("Status", "Low Risk", delta="Good", delta_color="normal")
            else:
                st.metric("Status", "Elevated", delta="Caution", delta_color="inverse")
        
        st.markdown("---")
        
        if p_l < threshold:
            st.success(f"✅ **Low Risk** (Score: {p_l*100:.1f}%)")
            st.write("### Recommendations:")
            st.write("- ✓ Continue maintaining healthy habits")
            st.write("- ✓ Regular check-ups every 1-2 years")
            st.write("- ✓ Focus on preventive care")
            st.write("- ✓ Monitor cholesterol and blood pressure")
        else:
            st.warning(f"⚠️ **Elevated Risk** (Score: {p_l*100:.1f}%)")
            st.write("### Next Steps:")
            st.write("- ⚠️ Proceed to **Stage 2: Clinical Diagnosis** for comprehensive assessment")
            st.write("- ⚠️ Consult with healthcare provider")
            st.write("- ⚠️ Consider lifestyle modifications")
            st.write("- ⚠️ Monitor vital signs regularly")
        
        # FIX: Create display dataframe with string values
        with st.expander("📋 Your Input Summary"):
            input_df = pd.DataFrame({
                'Parameter': ['Age', 'Sex', 'Total Cholesterol', 'LDL', 'HDL', 
                             'Systolic BP', 'Diastolic BP', 'Smoking', 'Diabetes'],
                'Value': [
                    str(age),
                    sex,
                    f"{total_cholesterol} mg/dL",
                    f"{ldl} mg/dL",
                    f"{hdl} mg/dL",
                    f"{systolic_bp} mm Hg",
                    f"{diastolic_bp} mm Hg",
                    smoking,
                    diabetes
                ]
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)

# ============================================================
# STAGE 2: CLINICAL DIAGNOSIS
# ============================================================
else:
    st.header("🏥 Stage 2: Clinical Diagnosis")
    st.warning("⚠️ This stage requires clinical measurements. Proceed only if Stage 1 indicated elevated risk.")
    
    # Get P_L from session state
    st.sidebar.markdown("---")
    if 'p_l' in st.session_state:
        default_p_l = st.session_state.p_l * 100
        st.sidebar.success(f"✓ Stage 1 Score loaded: {default_p_l:.1f}%")
    else:
        default_p_l = 65.0
        st.sidebar.info("Enter Stage 1 score manually or run Stage 1 first")
    
    p_l_input = st.sidebar.number_input("Stage 1 Risk Score (%)", 0.0, 100.0, default_p_l) / 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        clin_age = st.number_input("Age (years)", 18, 100, 50, key="clin_age")
        clin_sex = st.selectbox("Sex", ["Female", "Male"], key="clin_sex")
        
        st.subheader("Chest Pain Assessment")
        chest_pain_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
        chest_pain = st.selectbox("Chest Pain Type", chest_pain_options)
        
        st.subheader("Vital Signs")
        bp = st.number_input("Blood Pressure (mm Hg)", 80, 200, 120, key="clin_bp")
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200, key="clin_chol")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        
    with col2:
        st.subheader("Cardiac Tests")
        ekg_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        ekg = st.selectbox("EKG Results", ekg_options)
        
        max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        
        st.subheader("Advanced Measurements")
        st_depression = st.number_input("ST Depression (mm)", 0.0, 6.0, 0.0, step=0.1)
        slope_options = ["Upsloping", "Flat", "Downsloping"]
        slope = st.selectbox("Slope of Peak Exercise ST Segment", slope_options)
        
        vessels = st.selectbox("Number of Major Vessels (Fluoroscopy)", [0, 1, 2, 3])
        thallium_options = ["Normal", "Fixed Defect", "Reversible Defect"]
        thallium = st.selectbox("Thallium Stress Test", thallium_options)
    
    if st.button("Calculate Final Risk", type="primary"):
        # Encode inputs
        sex_encoded = 1 if clin_sex == "Male" else 0
        
        chest_pain_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
        chest_pain_encoded = chest_pain_map[chest_pain]
        
        fbs_encoded = 1 if fbs == "Yes" else 0
        
        ekg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        ekg_encoded = ekg_map[ekg]
        
        angina_encoded = 1 if exercise_angina == "Yes" else 0
        
        slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
        slope_encoded = slope_map[slope]
        
        thallium_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
        thallium_encoded = thallium_map[thallium]
        
        # FIX: Create DataFrame with proper feature names
        clinical_data = pd.DataFrame([[
            clin_age,
            sex_encoded,
            chest_pain_encoded,
            bp,
            cholesterol,
            fbs_encoded,
            ekg_encoded,
            max_hr,
            angina_encoded,
            st_depression,
            slope_encoded,
            vessels,
            thallium_encoded
        ]], columns=feature_info['clinical_features'])
        
        # Scale and predict
        clinical_scaled = scaler_clin.transform(clinical_data)
        p_c = clinical_model.predict_proba(clinical_scaled)[0][1]
        
        # Final fusion
        alpha = 0.4
        p_final = alpha * p_l_input + (1 - alpha) * p_c
        
        # Display results
        st.markdown("---")
        st.subheader("Comprehensive Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Lifestyle Risk", f"{p_l_input*100:.1f}%", 
                     help="From Stage 1 screening")
        with col2:
            st.metric("Clinical Risk", f"{p_c*100:.1f}%",
                     help="From Stage 2 diagnosis")
        with col3:
            st.metric("**Final Risk**", f"{p_final*100:.1f}%", 
                     delta=f"{(p_final-p_l_input)*100:+.1f}%",
                     help="Weighted combination (40% lifestyle + 60% clinical)")
        
        # Final gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_final * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Final Heart Disease Risk (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': p_final * 100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk interpretation
        st.markdown("---")
        st.subheader("Risk Assessment & Recommendations")
        
        if p_final < 0.3:
            st.success("✅ **Low Risk** - Continue regular preventive care")
            st.write("**Recommendations:**")
            st.write("- Annual health check-ups")
            st.write("- Maintain healthy lifestyle")
            st.write("- Monitor blood pressure and cholesterol")
        elif p_final < 0.6:
            st.warning("⚠️ **Moderate Risk** - Enhanced monitoring recommended")
            st.write("**Recommendations:**")
            st.write("- Consult cardiologist within 3 months")
            st.write("- Quarterly health monitoring")
            st.write("- Consider lifestyle interventions")
            st.write("- May need medication evaluation")
        else:
            st.error("🚨 **High Risk** - Immediate medical consultation advised")
            st.write("**Urgent Recommendations:**")
            st.write("- Schedule immediate appointment with cardiologist")
            st.write("- Comprehensive cardiac workup needed")
            st.write("- Likely medication required")
            st.write("- Intensive lifestyle modification program")
        
        # Breakdown visualization
        st.markdown("---")
        st.subheader("Risk Component Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(name='Contribution', 
                       x=['Lifestyle Risk', 'Clinical Risk'], 
                       y=[p_l_input*100*alpha, p_c*100*(1-alpha)],
                       marker_color=['#3498db', '#e74c3c'],
                       text=[f"{p_l_input*100*alpha:.1f}%", f"{p_c*100*(1-alpha):.1f}%"],
                       textposition='auto')
            ])
            fig.update_layout(
                title="Weighted Risk Contributions",
                yaxis_title="Contribution (%)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Risk Fusion Formula:**")
            st.code(f"Final = 0.4 × {p_l_input*100:.1f}% + 0.6 × {p_c*100:.1f}%")
            st.code(f"Final = {p_l_input*100*alpha:.1f}% + {p_c*100*(1-alpha):.1f}%")
            st.code(f"Final = {p_final*100:.1f}%")
            
            st.write("**Weight Distribution:**")
            st.write(f"- Lifestyle: 40% ({p_l_input*100*alpha:.1f}% absolute)")
            st.write(f"- Clinical: 60% ({p_c*100*(1-alpha):.1f}% absolute)")

# Footer
st.markdown("---")
st.caption("⚠️ **Medical Disclaimer**: This tool is for educational and screening purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info("""
**How it works:**
1. **Stage 1** screens using basic health profile
2. If risk ≥ 60%, proceed to **Stage 2**
3. **Stage 2** combines both assessments
4. Get personalized risk score and recommendations
""")