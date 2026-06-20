import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class SleepDiagnosticInference:
    def __init__(self, model_path, reference_df):
        self.model_path = model_path
        self.reference_df = reference_df
        self.model = self._load_model()
        self.target_cols = ['OSA_Risk', 'Insomnia_Risk', 'Sleep_Fragmentation', 'REM_Disruption']
        self.feature_cols = ['Age', 'Gender', 'Sleep duration', 'Sleep efficiency', 
                             'REM sleep percentage', 'Deep sleep percentage', 'Light sleep percentage']
        
    def _load_model(self):
        with open(self.model_path, 'rb') as file:
            return pickle.load(file)
            
    def _preprocess(self, input_data):
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        else:
            df_input = input_data.copy()
        le_gender = LabelEncoder()
        le_gender.fit(self.reference_df['Gender'])
        df_input['Gender'] = le_gender.transform(df_input['Gender'])
        return df_input[self.feature_cols].astype(float)

    def predict(self, raw_input):
        processed_input = self._preprocess(raw_input)
        raw_preds = self.model.predict(processed_input)[0]
        report = {}
        for i, target in enumerate(self.target_cols):
            le_target = LabelEncoder()
            le_target.fit(self.reference_df[target])
            report[target] = le_target.inverse_transform([int(raw_preds[i])])[0]
        return report

    def generate_report(self, raw_input):
        results = self.predict(raw_input)
        is_healthy = all('Low' in str(val) for val in results.values())
        print("--- Diagnostic Inference Report ---")
        print(f"Overall Status: {'HEALTHY' if is_healthy else 'RISK DETECTED'}")
        for risk, level in results.items():
            print(f"- {risk:20}: {level}")
        return results