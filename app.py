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
        
        /* Midnight Dark Dashboard Styling */
        .stApp {
            background-color: #0f172a;
            color: #f8fafc;
        }
        
        div[data-testid="metric-container"] {
            background: #1e293b;
            border: 1px solid #334155;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            border: 1px solid #334155;
            background-color: #1e293b;
            color: #f8fafc;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        /* Expander styling */
        div[data-testid="stExpander"] {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            color: #f8fafc;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            margin-bottom: 12px;
        }
        
        /* Headers */
        h1, h2, h3, p, label {
            color: #f8fafc !important;
            font-family: 'Inter', sans-serif;
        }
        
        hr {
            border-color: #334155;
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

    # Create Clinical Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview & Predictions", "Explainable AI (SHAP)", "Historical Population Data", "Batch Processing"])
    
    with tab1:
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
            
        st.subheader("Patient Vitals")
        
        # Row 1: KPI Cards
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Sleep", f"{sleep_duration} hrs")
        mcol2.metric("Efficiency", f"{int(sleep_efficiency * 100)}%")
        if sleep_efficiency > 0:
            time_in_bed = sleep_duration / sleep_efficiency
            wake_time = time_in_bed - sleep_duration
            mcol3.metric("Est. Wake Time", f"{wake_time:.1f} hrs")
        else:
            mcol3.metric("Est. Wake Time", "N/A")
            
        st.write("")
        st.subheader("Sleep Architecture")
        
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            # Sleep Efficiency Gauge
            fig_eff = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sleep_efficiency * 100,
                title = {'text': "Quality Score", 'font': {'color': '#f8fafc'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': '#f8fafc'},
                    'bar': {'color': "#f8fafc"},
                    'steps': [
                        {'range': [0, 79.9], 'color': "rgba(239, 68, 68, 0.6)"}, # Neon red
                        {'range': [79.9, 84.9], 'color': "rgba(245, 158, 11, 0.6)"}, # Amber
                        {'range': [84.9, 100], 'color': "rgba(16, 185, 129, 0.6)"} # Green
                    ],
                }
            ))
            fig_eff.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"})
            st.plotly_chart(fig_eff, use_container_width=True)
            
        with fig_col2:
            # Sleep Stages Donut
            stages = ['Deep Sleep', 'Light Sleep', 'REM Sleep']
            values = [deep_sleep, light_sleep, rem_sleep]
            fig_stages = go.Figure(data=[go.Pie(labels=stages, values=values, hole=.7, marker_colors=['#3b82f6', '#8b5cf6', '#06b6d4'])])
            fig_stages.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"})
            st.plotly_chart(fig_stages, use_container_width=True)
            
        st.divider()
        st.subheader("Diagnostic Risk Spectrum")
        st.write("Outputs from the MultiOutputClassifier indicating predicted probabilities and raw classes for all targets simultaneously.")
        
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
                colors_list.append('#f59e0b') # Amber
            elif info['score'] == 2:
                colors_list.append('#ef4444') # Red
            else:
                colors_list.append('#991b1b') # Dark red
                
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
            yaxis=dict(tickvals=[0, 1, 2, 3], ticktext=['Low Risk', 'Moderate', 'High', 'Severe'], range=[0, 3.5], gridcolor="#334155"),
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#f8fafc"}
        )
        st.plotly_chart(fig_risks, use_container_width=True)
        
        # Detailed Risk Explanations
        st.write("#### Clinical Insights")
        for target_col, info in risk_scores.items():
            if info['score'] > 0:
                with st.expander(f"⚠️ {target_col.replace('_', ' ').title()} - {info['label']}"):
                    st.write(f"{explanations_text[target_col]}")
            else:
                with st.expander(f"✅ {target_col.replace('_', ' ').title()} - Low Risk"):
                    st.write("No significant risk factors identified.")
                    
        max_score = max([info['score'] for info in risk_scores.values()])
        
        if max_score > 0:
            most_severe_risks = [disorder for disorder, info in risk_scores.items() if info['score'] == max_score]
            risk_names = ", ".join([f"{r.replace('_', ' ')}" for r in most_severe_risks])
            st.error(f"⚠️ **Medical Consultation Recommended**\n\nBased on your metrics, your most prominent risk factor is {risk_names}. Please consult with a specialist.")
        else:
            st.success("🎉 **All Clear**\n\nYour predicted sleep disorder risks are low. Keep up the good work!")
            
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
            
            pdf_bytes = generate_pdf_report(report_patient_data, risk_scores, explanations_text, max_score, risk_names if max_score > 0 else "")
            
            st.download_button(
                label="📄 Download Diagnostic PDF Report",
                data=pdf_bytes,
                file_name=f"lucidepoch_report_{report_patient_data['patient_id']}.pdf",
                mime="application/pdf",
                type="primary"
            )
        except Exception as e:
            st.error(f"Could not generate PDF summary: {e}")

    with tab2:
        st.subheader("Explainable AI (SHAP) Matrix")
        st.write("Querying the `shap.TreeExplainer` to isolate exact feature impacts for a specific target prediction.")
        
        selected_target = st.selectbox("Select condition to analyze:", target_cols)
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
                else:
                    shap_vals_to_plot = shap_values[-1][0]
            elif len(shap_values.shape) == 3:
                # Shape is (num_samples, num_features, num_classes)
                if pred_numeric < shap_values.shape[2]:
                    shap_vals_to_plot = shap_values[0, :, pred_numeric]
                else:
                    shap_vals_to_plot = shap_values[0, :, -1]
            else:
                shap_vals_to_plot = shap_values[0]
                
            predicted_label = le_targets[selected_target].inverse_transform([pred_numeric])[0]
            st.markdown(f"**Feature Impact for predicting '{predicted_label}' in {selected_target.replace('_', ' ')}:**")
            
            shap_df = pd.DataFrame({
                'Feature': feature_cols,
                'SHAP Value': shap_vals_to_plot,
                'Feature Value': input_df.iloc[0].values
            })
            
            # Sort by absolute impact for better visualization
            shap_df['Abs Value'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values(by='Abs Value', ascending=True)
            
            # Colors: Neon Red for increasing probability, Neon Blue for decreasing
            bar_colors = ['#ef4444' if val > 0 else '#3b82f6' for val in shap_df['SHAP Value']]
            
            hover_text = [
                f"Feature: {row['Feature']}<br>Value: {row['Feature Value']}<br>Impact: {row['SHAP Value']:.4f}"
                for _, row in shap_df.iterrows()
            ]
            
            fig_shap = go.Figure(go.Bar(
                x=shap_df['SHAP Value'],
                y=shap_df['Feature'],
                orientation='h',
                marker_color=bar_colors,
                text=shap_df['SHAP Value'].round(3),
                textposition='auto',
                hovertext=hover_text,
                hoverinfo='text'
            ))
            
            fig_shap.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#f8fafc"},
                xaxis=dict(gridcolor="#334155", zerolinecolor="#f8fafc", zerolinewidth=2, title="Impact on Prediction (SHAP Value)"),
                yaxis=dict(gridcolor="#334155")
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Red bars push the probability of this specific outcome higher; blue bars push it lower.")
        except Exception as e:
            st.error(f"Could not generate SHAP explanation: {e}")

    with tab3:
        st.subheader("Historical Population Demographics")
        st.write("Visualizations of the clinical dataset used to train the LucidEpoch framework.")
        
        if not hist_data.empty:
            dem_col1, dem_col2 = st.columns(2)
            
            with dem_col1:
                # Age Distribution
                fig_age = px.histogram(hist_data, x="Age", title="Age Distribution of Training Cohort", color_discrete_sequence=['#3b82f6'])
                fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"})
                st.plotly_chart(fig_age, use_container_width=True)
                
            with dem_col2:
                # Gender Distribution
                if 'Gender' in hist_data.columns:
                    gender_counts = hist_data['Gender'].value_counts()
                    fig_gender = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values, hole=.4, marker_colors=['#8b5cf6', '#10b981'])])
                    fig_gender.update_layout(title_text="Gender Breakdown", paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"})
                    st.plotly_chart(fig_gender, use_container_width=True)
                
            st.divider()
            
            if 'OSA_Risk' in hist_data.columns:
                osa_counts = hist_data['OSA_Risk'].value_counts().reset_index()
                osa_counts.columns = ['Risk Level', 'Count']
                fig_osa = px.bar(osa_counts, x='Risk Level', y='Count', title="Training Data: OSA Risk Labels", color='Risk Level',
                                color_discrete_map={'Low OSA Risk': '#10b981', 'Moderate OSA Risk': '#f59e0b', 'High OSA Risk': '#ef4444', 'Severe OSA Risk': '#991b1b', 'Low': '#10b981', 'Moderate': '#f59e0b', 'High': '#ef4444'})
                fig_osa.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"})
                st.plotly_chart(fig_osa, use_container_width=True)
            
            with st.expander("📥 Download Raw Dataset (CSV)"):
                st.write("For verification purposes, you can download the raw dataset used to train the model.")
                csv = hist_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download historical_data.csv",
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
