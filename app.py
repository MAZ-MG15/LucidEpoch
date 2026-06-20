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

st.markdown("""
    <style>
        /* Glassmorphism premium styling */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f1f5f9;
        }
        
        div[data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
            border-color: rgba(0, 242, 254, 0.3);
        }
        
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Expander styling */
        div[data-testid="stExpander"] {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
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

st.image("banner.png", use_container_width=True)

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
    st.sidebar.header("Wearable Data Simulator")
    
    default_age = st.session_state.get('age', 30)
    default_gender = st.session_state.get('gender', "Male")
    default_gender_idx = 0 if default_gender == "Male" else 1
    
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=default_age)
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"], index=default_gender_idx)
    sleep_duration = st.sidebar.slider("Sleep duration (hours)", min_value=0.0, max_value=24.0, value=st.session_state.get('sleep_duration', 7.0), step=0.1)
    sleep_efficiency = st.sidebar.slider("Sleep efficiency", min_value=0.0, max_value=1.0, value=st.session_state.get('sleep_efficiency', 0.85), step=0.01)
    rem_sleep = st.sidebar.slider("REM sleep percentage (%)", min_value=0, max_value=100, value=st.session_state.get('rem_sleep', 20))
    deep_sleep = st.sidebar.slider("Deep sleep percentage (%)", min_value=0, max_value=100, value=st.session_state.get('deep_sleep', 20))
    light_sleep = st.sidebar.slider("Light sleep percentage (%)", min_value=0, max_value=100, value=st.session_state.get('light_sleep', 60))
    
    st.sidebar.divider()
    
    total_sleep_stages = rem_sleep + deep_sleep + light_sleep
    if total_sleep_stages != 100:
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

    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview & Predictions", "Explainable AI (SHAP)", "Historical Population Data", "Batch Processing"])
    
    with tab1:
        st.subheader("Current Wearable Metrics")
        
        # Plotly Gauges and Charts
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # Sleep Efficiency Gauge
            fig_eff = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sleep_efficiency * 100,
                title = {'text': "Sleep Score (Efficiency)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00f2fe"},
                    'steps': [
                        {'range': [0, 60], 'color': "rgba(231, 76, 60, 0.2)"},
                        {'range': [60, 85], 'color': "rgba(243, 156, 18, 0.2)"},
                        {'range': [85, 100], 'color': "rgba(46, 204, 113, 0.2)"}
                    ],
                }
            ))
            fig_eff.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f1f5f9"})
            st.plotly_chart(fig_eff, use_container_width=True)
            
        with fig_col2:
            # Sleep Stages Donut
            stages = ['Deep Sleep', 'Light Sleep', 'REM Sleep']
            values = [deep_sleep, light_sleep, rem_sleep]
            fig_stages = go.Figure(data=[go.Pie(labels=stages, values=values, hole=.6, marker_colors=['#3b82f6', '#8b5cf6', '#10b981'])])
            fig_stages.update_layout(title_text="Sleep Stages Breakdown", height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f1f5f9"})
            st.plotly_chart(fig_stages, use_container_width=True)
            
        # Quick metrics row underneath
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Total Sleep Duration", f"{sleep_duration} hrs")
        mcol2.metric("Patient Profile", f"{age} yrs, {gender}")
        
        st.divider()
        st.subheader("Diagnostic Risk Predictions")
        
        if predict_clicked or 'predictions' not in st.session_state:
            prediction = model.predict(input_df)
            st.session_state['predictions'] = prediction
        else:
            prediction = st.session_state['predictions']
            
        target_cols = list(le_targets.keys())
        cols = st.columns(len(target_cols))
        
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
            
        # Consolidated Plotly Risk Bar Chart
        risk_names_list = []
        risk_scores_list = []
        colors_list = []
        
        for target_col, info in risk_scores.items():
            risk_names_list.append(target_col.replace('_', ' ').title())
            risk_scores_list.append(info['score'])
            if info['score'] == 0:
                colors_list.append('#10b981') # Green
            elif info['score'] == 1:
                colors_list.append('#f59e0b') # Yellow
            elif info['score'] == 2:
                colors_list.append('#ef4444') # Red
            else:
                colors_list.append('#991b1b') # Dark Red
                
        fig_risks = go.Figure(data=[
            go.Bar(
                x=risk_names_list, 
                y=risk_scores_list,
                marker_color=colors_list,
                text=[info['label'] for info in risk_scores.values()],
                textposition='auto',
                width=0.5
            )
        ])
        fig_risks.update_layout(
            title="Diagnostic Risk Spectrum",
            yaxis=dict(tickvals=[0, 1, 2, 3], ticktext=['Low', 'Moderate', 'High', 'Severe'], range=[0, 3.5]),
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#f1f5f9"}
        )
        st.plotly_chart(fig_risks, use_container_width=True)
        
        st.subheader("Detailed Risk Explanations")
        for target_col, info in risk_scores.items():
            if info['score'] > 0:
                with st.expander(f"⚠️ {target_col.replace('_', ' ').title()} - {info['label']}"):
                    st.write(f"**Why?** {explanations_text[target_col]}")
            else:
                with st.expander(f"✅ {target_col.replace('_', ' ').title()} - Low Risk"):
                    st.write("No significant risk factors identified for this condition.")
                    
        st.divider()
        max_score = max([info['score'] for info in risk_scores.values()])
        
        risk_names = ""
        if max_score > 0:
            most_severe_risks = [disorder for disorder, info in risk_scores.items() if info['score'] == max_score]
            risk_names = ", ".join([f"**{r.replace('_', ' ')}**" for r in most_severe_risks])
            st.error(f"⚠️ **Medical Consultation Recommended**\n\nBased on your metrics, your most prominent risk factor is {risk_names}. We strongly recommend consulting with a doctor or sleep specialist for a professional diagnosis and guidance.")
        else:
            st.success("🎉 **Great News!**\n\nYour predicted sleep disorder risks are low. Keep up the good sleep hygiene!")
            
        # Report Generation
        st.subheader("Download Your Diagnostic Report")
        st.write("Get a comprehensive professional PDF summary of your predictions and factors.")
        
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
            
            pdf_bytes = generate_pdf_report(report_patient_data, risk_scores, explanations_text, max_score, risk_names)
            
            st.download_button(
                label="📄 Download Diagnostic PDF Report",
                data=pdf_bytes,
                file_name=f"lucidepoch_report_{report_patient_data['patient_id']}.pdf",
                mime="application/pdf",
                type="secondary"
            )
        except Exception as e:
            st.error(f"Could not generate PDF report: {e}")
            
    with tab2:
        st.subheader("Explainable AI (XAI) Simulator")
        st.write("Understand the specific factors driving each risk prediction. We query the SHAP explainer to see how your physiological features push your risk up or down.")
        
        st.info("💡 **How to read this chart:** \n\n- **Red bars** represent factors that increase the probability of this risk.\n- **Blue bars** represent factors that decrease the probability of this risk.\n- The size of the bar indicates the magnitude of the impact.")
        
        selected_target = st.selectbox("Select sleep condition to analyze:", target_cols)
        target_idx = target_cols.index(selected_target)
        
        # MultiOutputClassifier wraps individual classifiers
        single_model = model.estimators_[target_idx]
        
        try:
            explainer = shap.TreeExplainer(single_model)
            shap_values = explainer.shap_values(input_df)
            
            # Use the prediction index to show the explanation for the predicted class
            pred_numeric = prediction[0][target_idx]
            
            if isinstance(shap_values, list):
                if pred_numeric < len(shap_values):
                    shap_vals_to_plot = shap_values[pred_numeric][0]
                    expected_value = explainer.expected_value[pred_numeric]
                else:
                    shap_vals_to_plot = shap_values[-1][0]
                    expected_value = explainer.expected_value[-1]
            elif len(shap_values.shape) == 3:
                # Shape is (num_samples, num_features, num_classes)
                if pred_numeric < shap_values.shape[2]:
                    shap_vals_to_plot = shap_values[0, :, pred_numeric]
                    expected_value = explainer.expected_value[pred_numeric]
                else:
                    shap_vals_to_plot = shap_values[0, :, -1]
                    expected_value = explainer.expected_value[-1]
            else:
                shap_vals_to_plot = shap_values[0]
                expected_value = explainer.expected_value
                
            predicted_label = le_targets[selected_target].inverse_transform([pred_numeric])[0]
            st.markdown(f"**Feature Impact for predicting '{predicted_label}' in {selected_target.replace('_', ' ')}:**")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.force(
                expected_value, 
                shap_vals_to_plot, 
                features=input_df.iloc[0], 
                feature_names=feature_cols,
                matplotlib=True,
                show=False
            )
            st.pyplot(fig)
            st.caption("Red bars push the probability of this specific outcome higher; blue bars push it lower.")
        except Exception as e:
            st.error(f"Could not generate SHAP explanation: {e}")

    with tab3:
        st.subheader("Historical Population Data")
        st.write("A scalable view of the raw physiological metrics used to train the Multi-Output Random Forest.")
        
        if not hist_data.empty:
            st.dataframe(hist_data.head(100), use_container_width=True)
            
            csv = hist_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Historical Data as CSV",
                data=csv,
                file_name='lucidepoch_historical_data.csv',
                mime='text/csv',
            )
        else:
            st.warning("Historical data not found.")

    with tab4:
        st.subheader("Batch Patient Processing")
        st.write("Upload a CSV file containing multiple patients to generate predictions in bulk. This is ideal for processing daily clinic intake.")
        
        uploaded_file = st.file_uploader("Upload Patient CSV", type=['csv'])
        st.info("Required Columns: Age, Gender, Sleep duration, Sleep efficiency, REM sleep percentage, Deep sleep percentage, Light sleep percentage")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_df.head())
                
                if st.button("Process Batch Predictions", type="primary"):
                    with st.spinner("Processing patients..."):
                        process_df = batch_df.copy()
                        # Encode gender
                        process_df['Gender'] = le_gender.transform(process_df['Gender'])
                        process_df = process_df[feature_cols]
                        
                        batch_preds = model.predict(process_df)
                        
                        target_cols = list(le_targets.keys())
                        for i, target_col in enumerate(target_cols):
                            batch_df[f'{target_col}_Risk'] = le_targets[target_col].inverse_transform(batch_preds[:, i])
                            
                        st.success(f"Batch processing complete for {len(batch_df)} patients!")
                        st.dataframe(batch_df)
                        
                        csv_output = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Batch Results (CSV)",
                            data=csv_output,
                            file_name="batch_predictions_results.csv",
                            mime="text/csv",
                            type="secondary"
                        )
            except Exception as e:
                st.error(f"Error processing batch file. Please ensure it matches the required format. Details: {e}")

else:
    st.warning("Model files not found. Please run `train_model.py` first to generate the models.")
