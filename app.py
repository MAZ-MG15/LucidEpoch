import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sleep Disorder Risk Predictor", layout="wide", page_icon="🌙")

st.markdown("""
    <style>
        /* Card styling for metrics */
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 4px 4px 10px rgba(0,0,0,0.1);
        }
        
        /* Make prediction boxes pop */
        div[data-testid="stAlert"] {
            border-radius: 8px !important;
            box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        div[data-testid="stAlert"]:hover {
            transform: translateY(-2px);
            box-shadow: 4px 4px 10px rgba(0,0,0,0.1);
        }
        
        /* Dark mode support adjustments */
        @media (prefers-color-scheme: dark) {
            div[data-testid="metric-container"] {
                background-color: #1e1e1e;
                border-color: #333;
            }
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

st.title("LucidEpoch: Clinical Sleep Assessment Dashboard")
st.write("A machine learning framework for assessing multi-output sleep disorder risks using wearable data.")

if model is not None:
    st.sidebar.header("Wearable Data Simulator")
    st.sidebar.write("Adjust the physiological metrics below to see how they impact predictions.")
    
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30, help="Patient's age in years.")
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"], help="Biological gender of the patient.")
    sleep_duration = st.sidebar.slider("Sleep duration (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.1, help="Total hours of sleep per day.")
    sleep_efficiency = st.sidebar.slider("Sleep efficiency", min_value=0.0, max_value=1.0, value=0.85, step=0.01, help="Percentage of time spent asleep while in bed (0.0 to 1.0). Higher is better.")
    rem_sleep = st.sidebar.slider("REM sleep percentage (%)", min_value=0, max_value=100, value=20, help="Percentage of sleep spent in the REM (Rapid Eye Movement) stage. Important for cognitive function.")
    deep_sleep = st.sidebar.slider("Deep sleep percentage (%)", min_value=0, max_value=100, value=20, help="Percentage of sleep spent in the Deep sleep stage. Crucial for physical restoration.")
    light_sleep = st.sidebar.slider("Light sleep percentage (%)", min_value=0, max_value=100, value=60, help="Percentage of sleep spent in the Light sleep stage.")
    
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
    tab1, tab2, tab3 = st.tabs(["Overview & Predictions", "Explainable AI (SHAP)", "Historical Population Data"])
    
    with tab1:
        st.subheader("Current Wearable Metrics")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Sleep Efficiency", f"{int(sleep_efficiency*100)}%")
        mcol2.metric("Deep Sleep", f"{deep_sleep}%")
        mcol3.metric("Sleep Duration", f"{sleep_duration} hrs")
        mcol4.metric("REM Sleep", f"{rem_sleep}%")
        
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
            
            with cols[i]:
                if score == 0:
                    st.success(f"**{target_col.replace('_', ' ')}**\n\n{pred_label}")
                elif score == 1:
                    st.warning(f"**{target_col.replace('_', ' ')}**\n\n{pred_label}\n\n*Why?* {explanation}")
                else:
                    st.error(f"**{target_col.replace('_', ' ')}**\n\n{pred_label}\n\n*Why?* {explanation}")
                    
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
        st.write("Get a comprehensive summary of your predictions and factors.")
        
        report_lines = []
        report_lines.append("LUCIDEPOCH CLINICAL SLEEP ASSESSMENT REPORT")
        report_lines.append("===========================================\n")
        report_lines.append("PATIENT METRICS:")
        report_lines.append(f"- Age: {age}")
        report_lines.append(f"- Gender: {gender}")
        report_lines.append(f"- Sleep duration: {sleep_duration} hrs")
        report_lines.append(f"- Sleep efficiency: {int(sleep_efficiency*100)}%")
        report_lines.append(f"- REM sleep: {rem_sleep}%")
        report_lines.append(f"- Deep sleep: {deep_sleep}%")
        report_lines.append(f"- Light sleep: {light_sleep}%\n")
        
        report_lines.append("DIAGNOSTIC RISK PREDICTIONS:")
        for target_col, info in risk_scores.items():
            report_lines.append(f"- {target_col.replace('_', ' ').upper()}: {info['label']}")
            if info['score'] > 0 and explanations_text[target_col]:
                clean_exp = explanations_text[target_col].replace('**', '')
                report_lines.append(f"  Reason: {clean_exp}")
        
        report_lines.append("\nRECOMMENDATION:")
        if max_score > 0:
            report_lines.append("Medical Consultation Recommended.")
            report_lines.append(f"Based on your metrics, your most prominent risk factor is {risk_names.replace('**', '')}.")
            report_lines.append("We strongly recommend consulting with a doctor or sleep specialist for a professional diagnosis and guidance.")
        else:
            report_lines.append("Your predicted sleep disorder risks are low. Keep up the good sleep hygiene!")
            
        report_text = "\n".join(report_lines)
        
        st.download_button(
            label="📄 Download Diagnostic Report",
            data=report_text,
            file_name="lucidepoch_diagnostic_report.txt",
            mime="text/plain",
            type="secondary"
        )
            
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

else:
    st.warning("Model files not found. Please run `train_model.py` first to generate the models.")
