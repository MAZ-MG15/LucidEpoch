import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os

df = pd.read_csv('LucidEpoch_Final_Dataset.csv')
print("Dataset loaded! Shape:", df.shape)

# Encode Gender
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

feature_cols = ['Age', 'Gender', 'Sleep duration', 'Sleep efficiency',
                'REM sleep percentage', 'Deep sleep percentage', 'Light sleep percentage']

X = df[feature_cols]
target_cols = ['OSA_Risk', 'Insomnia_Risk', 'Sleep_Fragmentation', 'REM_Disruption']

# Encode targets
le_targets = {}
for col in target_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_targets[col] = le

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, df[target_cols], test_size=0.2, random_state=42)

# Train Model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

# Save Model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/lucid_epoch_model.pkl')
joblib.dump(le_gender, 'model/gender_encoder.pkl')
joblib.dump(le_targets, 'model/target_encoders.pkl')
joblib.dump(feature_cols, 'model/feature_names.pkl')

print("Model trained and saved successfully!")