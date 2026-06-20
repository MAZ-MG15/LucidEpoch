import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

model_dir = 'model'
model = joblib.load(os.path.join('MODELS', 'sleep_disorder_balanced_model.pkl'))
le_gender = joblib.load(os.path.join(model_dir, 'gender_encoder.pkl'))
le_targets = joblib.load(os.path.join(model_dir, 'target_encoders.pkl'))
feature_cols = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))

input_data = {
    'Age': 30,
    'Gender': 'Male',
    'Sleep duration': 7.0,
    'Sleep efficiency': 0.85,
    'REM sleep percentage': 20,
    'Deep sleep percentage': 20,
    'Light sleep percentage': 60
}

input_df = pd.DataFrame([input_data])
input_df['Gender'] = le_gender.transform(input_df['Gender'])
input_df = input_df[feature_cols]

prediction = model.predict(input_df)

target_cols = list(le_targets.keys())
target_idx = 0 # OSA Risk
single_model = model.estimators_[target_idx]

explainer = shap.TreeExplainer(single_model)
shap_values = explainer.shap_values(input_df)

pred_numeric = prediction[0][target_idx]

if isinstance(shap_values, list):
    if pred_numeric < len(shap_values):
        shap_vals_to_plot = shap_values[pred_numeric]
        expected_value = explainer.expected_value[pred_numeric]
    else:
        shap_vals_to_plot = shap_values[-1]
        expected_value = explainer.expected_value[-1]
else:
    shap_vals_to_plot = shap_values
    expected_value = explainer.expected_value

try:
    fig, ax = plt.subplots(figsize=(10, 4))
    res = shap.plots.force(
        expected_value, 
        shap_vals_to_plot[0], 
        features=input_df.iloc[0], 
        feature_names=feature_cols,
        matplotlib=True,
        show=False
    )
    print("SHAP plot generated successfully.")
except Exception as e:
    print(f"ERROR: {e}")
