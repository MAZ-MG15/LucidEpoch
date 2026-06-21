import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import db_utils
from pdf_utils import generate_pdf_report

db_utils.init_db()

st.set_page_config(page_title="Sleep Disorder Risk Predictor", layout="wide", page_icon="🌙")

st.warning("⚠️ **CLINICAL DISCLAIMER:** This dashboard is a research prototype developed for academic purposes. It is not a certified medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")

st.markdown("""
    <style>
        /* Hide Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Friendly Wellness Styling */
        .stApp {
            background-color: #faf9f6;
            color: #4b5563;
        }
        
        div[data-testid="metric-container"] {
            background: #ffffff;
            border: none;
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        }
        
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            border: none;
            background-color: #ffffff;
            color: #4b5563;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        }
        
        /* Expander styling */
        div[data-testid="stExpander"] {
            background: #ffffff;
            border: none;
            border-radius: 12px;
            color: #4b5563;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
            margin-bottom: 12px;
        }
        
        /* Headers */
        h1, h2, h3, p, label {
            color: #374151 !important;
            font-family: 'Inter', sans-serif;
        }
        
        hr {
            border-color: #f3f4f6;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_components():
    model_dir = 'model'
    try:
        model = joblib.load(os.path.join('MODELS', 'sleep_disorder_balanced_model.pkl'))
        le_gender = joblib.load(os.path.join(model_dir, 'gender_encoder.pkl'))
        le_targets = joblib.load(os.path.join(model_dir, 'target_encoders.pkl'))
        feature_cols = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        return model, le_gender, le_targets, feature_cols
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None, None

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv('LucidEpoch_Final_Dataset.csv')
        return df
    except Exception as e:
        return pd.DataFrame()

def get_simple_explanation(model, input_df, target_idx, feature_cols, pred_numeric):
    try:
        single_model = model.estimators_[target_idx]
        explainer = shap.TreeExplainer(single_model)
        shap_values = explainer.shap_values(input_df)
        
        if isinstance(shap_values, list):
            if pred_numeric < len(shap_values):
                shap_vals = shap_values[pred_numeric][0]
            else:
                shap_vals = shap_values[-1][0]
        elif len(shap_values.shape) == 3:
            if pred_numeric < shap_values.shape[2]:
                shap_vals = shap_values[0, :, pred_numeric]
            else:
                shap_vals = shap_values[0, :, -1]
        else:
            shap_vals = shap_values[0]
            
        feature_impacts = list(zip(feature_cols, shap_vals, input_df.iloc[0]))
        # Filter for features that increased the risk
        positive_impacts = [x for x in feature_impacts if x[1] > 0]
        positive_impacts.sort(key=lambda x: x[1], reverse=True)
        
        top_features = positive_impacts[:2]
        
        if not top_features:
            return "No specific dominant factors identified."
            
        explanations = []
        for feat_name, impact, feat_val in top_features:
            if feat_name == "Gender":
                val_str = "your gender"
            elif feat_name == "Sleep efficiency":
                val_str = f"{int(feat_val*100)}%"
            else:
                val_str = f"{feat_val}"
                
            explanations.append(f"**{feat_name.replace('_', ' ')}** ({val_str})")
            
        reason = " and ".join(explanations)
        return f"Main contributing factors: {reason}."
    except Exception as e:
        return "Explanation unavailable."

model, le_gender, le_targets, feature_cols = load_model_components()
hist_data = load_historical_data()

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("logo.png", width=80)
with col_title:
    st.title("LucidEpoch: Clinical Sleep Assessment Dashboard")
    
st.write("A machine learning framework for assessing multi-output sleep disorder risks using wearable data.")

st.sidebar.image("logo.png", width=120)
if model is not None:
    st.sidebar.header("Patient Record Management")
    
    patients_df = db_utils.get_all_patients()
    if not patients_df.empty:
        patient_list = ["-- New Patient --"] + patients_df['patient_id'].tolist()
        selected_patient_id = st.sidebar.selectbox("Load Patient Profile", options=patient_list)
        
        if selected_patient_id != "-- New Patient --":
            loaded_data = db_utils.get_patient(selected_patient_id)
            if loaded_data:
                st.session_state['age'] = loaded_data['age']
                st.session_state['gender'] = loaded_data['gender']
                st.session_state['sleep_duration'] = loaded_data['sleep_duration']
                st.session_state['sleep_efficiency'] = loaded_data['sleep_efficiency']
                st.session_state['rem_sleep'] = loaded_data['rem_sleep']
                st.session_state['deep_sleep'] = loaded_data['deep_sleep']
                st.session_state['light_sleep'] = loaded_data['light_sleep']
    
    st.sidebar.divider()
    st.sidebar.header("Connect Your Smartwatch")
    
    input_mode = st.sidebar.radio("Data Source", ["⌚ Sync Smartwatch (Recommended)", "🛠️ Manual Entry"])
    is_manual = "Manual Entry" in input_mode
    
    if not is_manual:
        st.sidebar.info("Select your smartwatch to securely sync your sleep data.")
        mock_device = st.sidebar.selectbox("Select Device", ["Apple Watch", "Fitbit", "Oura Ring", "Garmin"])
        if st.sidebar.button("🔄 Sync Sleep Data", use_container_width=True):
            if not hist_data.empty:
                import random
                rand_idx = random.randint(0, len(hist_data)-1)
                synced = hist_data.iloc[rand_idx]
                
                st.session_state['age'] = int(synced['Age'])
                st.session_state['gender'] = synced['Gender']
                st.session_state['sleep_duration'] = float(synced['Sleep duration'])
                st.session_state['sleep_efficiency'] = float(synced['Sleep efficiency'])
                
                # Ensure they sum to 100 exactly
                rem = float(synced['REM sleep percentage'])
                deep = float(synced['Deep sleep percentage'])
                light = float(synced['Light sleep percentage'])
                total = rem + deep + light
                if total > 0:
                    rem = (rem / total) * 100
                    deep = (deep / total) * 100
                
                st.session_state['rem_sleep'] = int(round(rem))
                st.session_state['deep_sleep'] = int(round(deep))
                st.session_state['light_sleep'] = 100 - st.session_state['rem_sleep'] - st.session_state['deep_sleep']
                
                st.session_state['sync_success'] = True
    else:
        st.sidebar.warning("⚠️ Manual Override active. For clinical stress-testing only.")
        st.session_state['sync_success'] = False

    st.sidebar.divider()
    
    default_age = st.session_state.get('age', 30)
    default_gender = st.session_state.get('gender', "Male")
    default_gender_idx = 0 if default_gender == "Male" else 1
    
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=default_age, disabled=not is_manual)
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"], index=default_gender_idx, disabled=not is_manual)
    sleep_duration = st.sidebar.slider("Sleep duration (hours)", min_value=0.0, max_value=24.0, value=st.session_state.get('sleep_duration', 7.0), step=0.1, disabled=not is_manual)
    sleep_efficiency = st.sidebar.slider("Sleep efficiency", min_value=0.0, max_value=1.0, value=st.session_state.get('sleep_efficiency', 0.85), step=0.01, disabled=not is_manual)
    rem_sleep = st.sidebar.slider("REM sleep percentage (Healthy: 20-25%)", min_value=0, max_value=100, value=st.session_state.get('rem_sleep', 25), disabled=not is_manual)
    deep_sleep = st.sidebar.slider("Deep sleep percentage (Healthy: 15-20%)", min_value=0, max_value=100, value=st.session_state.get('deep_sleep', 20), disabled=not is_manual)
    light_sleep = st.sidebar.slider("Light sleep percentage (Healthy: 45-55%)", min_value=0, max_value=100, value=st.session_state.get('light_sleep', 55), disabled=not is_manual)
    
    st.sidebar.divider()
    
    total_sleep_stages = rem_sleep + deep_sleep + light_sleep
    if is_manual and total_sleep_stages != 100:
        st.sidebar.error(f"⚠️ Invalid Input: Sleep stages currently sum to {total_sleep_stages}%. They must add up to exactly 100%.")
        st.error("Please adjust the sleep stage percentages in the sidebar to equal exactly 100% before viewing predictions.")
        st.stop()
        
    patient_id_input = st.sidebar.text_input("Patient ID (for saving)", placeholder="e.g., PT-001")
    if st.sidebar.button("💾 Save Patient Record"):
        if patient_id_input:
            patient_data = {
                'patient_id': patient_id_input,
                'age': age,
                'gender': gender,
                'sleep_duration': sleep_duration,
                'sleep_efficiency': sleep_efficiency,
                'rem_sleep': rem_sleep,
                'deep_sleep': deep_sleep,
                'light_sleep': light_sleep
            }
            success, msg = db_utils.save_patient(patient_data)
            if success:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(f"Error: {msg}")
        else:
            st.sidebar.warning("Please enter a Patient ID to save.")
            
    predict_clicked = st.sidebar.button("Predict Risks", type="primary")
    
    with st.sidebar.expander("📝 Methodology Note"):
        st.caption("This live prototype uses a simplified feature set (Age, Gender, Sleep Duration, Efficiency, and Stage Percentages) for inference. Features like *Awakenings* and *Alcohol Consumption* utilized in the Chapter 3.3 OSA derivation were excluded from this specific deployment build to maintain real-time stability with the serialized model.")

    input_data = {
        'Age': age,
        'Gender': gender,
        'Sleep duration': sleep_duration,
        'Sleep efficiency': sleep_efficiency,
        'REM sleep percentage': rem_sleep,
        'Deep sleep percentage': deep_sleep,
        'Light sleep percentage': light_sleep
    }
    
    input_df = pd.DataFrame([input_data])
    
    try:
        input_df['Gender'] = le_gender.transform(input_df['Gender'])
    except Exception as e:
        st.error(f"Error encoding gender: {e}")
        st.stop()
        
    input_df = input_df[feature_cols]

    # Simple Storybook Flow (No Tabs)
    if predict_clicked or 'predictions' not in st.session_state:
        prediction = model.predict(input_df)
        st.session_state['predictions'] = prediction
    else:
        prediction = st.session_state['predictions']
        
    target_cols = list(le_targets.keys())
    risk_scores = {}
    explanations_text = {}
    
    for i, target_col in enumerate(target_cols):
        pred_numeric = prediction[0][i]
        pred_label = le_targets[target_col].inverse_transform([pred_numeric])[0]
        
        formatted_label = str(pred_label).lower()
        score = 0
        if 'severe' in formatted_label or 'very high' in formatted_label:
            score = 3
        elif 'high' in formatted_label:
            score = 2
        elif 'moderate' in formatted_label:
            score = 1
            
        risk_scores[target_col] = {'label': pred_label, 'score': score}
        
        explanation = ""
        if score > 0:
            explanation = get_simple_explanation(model, input_df, i, feature_cols, pred_numeric)
        explanations_text[target_col] = explanation
        
    st.header("Good morning! ☀️")
    st.write("Here is a gentle look at how you slept last night.")
    
    # Row 1: KPI Cards
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Time Asleep", f"{sleep_duration} hrs")
    mcol2.metric("Sleep Quality", f"{int(sleep_efficiency * 100)}%")
    if sleep_efficiency > 0:
        time_in_bed = sleep_duration / sleep_efficiency
        wake_time = time_in_bed - sleep_duration
        mcol3.metric("Time Awake in Bed", f"{wake_time:.1f} hrs")
    else:
        mcol3.metric("Time Awake in Bed", "N/A")
        
    st.divider()
    st.subheader("Your Sleep Health Check 🌙")
    st.write("We analyzed your sleep patterns to check for any signs of common sleep disruptions.")
    
    # Consolidated Plotly Risk Bar Chart
    risk_names_list = []
    risk_scores_list = []
    colors_list = []
    
    for target_col, info in risk_scores.items():
        risk_names_list.append(target_col.replace('_', ' ').title())
        risk_scores_list.append(info['score'])
        if info['score'] == 0:
            colors_list.append('#86efac') # Soft mint green
        elif info['score'] == 1:
            colors_list.append('#fcd34d') # Soft yellow
        elif info['score'] == 2:
            colors_list.append('#fca5a5') # Soft red
        else:
            colors_list.append('#f87171') # Slightly darker soft red
            
    fig_risks = go.Figure(data=[
        go.Bar(
            x=risk_names_list, 
            y=risk_scores_list,
            marker_color=colors_list,
            text=[info['label'] for info in risk_scores.values()],
            textposition='auto',
            width=0.4
        )
    ])
    fig_risks.update_layout(
        yaxis=dict(tickvals=[0, 1, 2, 3], ticktext=['All Good', 'Slightly Elevated', 'High', 'Severe'], range=[0, 3.5], gridcolor="#f3f4f6"),
        height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#4b5563"}
    )
    st.plotly_chart(fig_risks, use_container_width=True)
    
    # Detailed Risk Explanations
    st.subheader("Why Did We Give You This Score?")
    for target_col, info in risk_scores.items():
        if info['score'] > 0:
            with st.expander(f"🌼 {target_col.replace('_', ' ').title()} - {info['label']}"):
                st.write(f"{explanations_text[target_col]}")
        else:
            with st.expander(f"✨ {target_col.replace('_', ' ').title()} - Looking Good!"):
                st.write("We didn't find any significant disruptions here. Keep up the great habits!")
                
    max_score = max([info['score'] for info in risk_scores.values()])
    
    if max_score > 0:
        most_severe_risks = [disorder for disorder, info in risk_scores.items() if info['score'] == max_score]
        risk_names = ", ".join([f"{r.replace('_', ' ')}" for r in most_severe_risks])
        st.error(f"🌱 **Gentle Reminder:** We noticed some signs of {risk_names}. It might be helpful to chat with a sleep specialist just to make sure you are getting the rest you deserve.")
    else:
        st.success("🎉 **Sweet Dreams!** Your sleep health is looking wonderful. Keep prioritizing your rest.")
        
    st.divider()
    st.subheader("Download Your Summary")
    st.write("Get a friendly PDF summary of your sleep health to save or share.")
    
    try:
        # Reconstruct patient_data for the report
        report_patient_data = {
            'patient_id': patient_id_input if 'patient_id_input' in locals() and patient_id_input else 'N/A',
            'age': age,
            'gender': gender,
            'sleep_duration': sleep_duration,
            'sleep_efficiency': sleep_efficiency,
            'rem_sleep': rem_sleep,
            'deep_sleep': deep_sleep,
            'light_sleep': light_sleep
        }
        
        pdf_bytes = generate_pdf_report(report_patient_data, risk_scores, explanations_text, max_score, risk_names if max_score > 0 else "")
        
        st.download_button(
            label="📄 Download PDF Summary",
            data=pdf_bytes,
            file_name=f"sleep_summary_{report_patient_data['patient_id']}.pdf",
            mime="application/pdf",
            type="primary"
        )
    except Exception as e:
        st.error(f"Could not generate PDF summary: {e}")

else:
    st.warning("Model files not found. Please run `train_model.py` first to generate the models.")
