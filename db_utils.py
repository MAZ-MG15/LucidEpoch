import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = 'patients.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            sleep_duration REAL,
            sleep_efficiency REAL,
            rem_sleep INTEGER,
            deep_sleep INTEGER,
            light_sleep INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_patient(patient_data):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Upsert logic (insert or replace)
        cursor.execute('''
            INSERT OR REPLACE INTO patients 
            (patient_id, age, gender, sleep_duration, sleep_efficiency, rem_sleep, deep_sleep, light_sleep, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_data['patient_id'],
            patient_data['age'],
            patient_data['gender'],
            patient_data['sleep_duration'],
            patient_data['sleep_efficiency'],
            patient_data['rem_sleep'],
            patient_data['deep_sleep'],
            patient_data['light_sleep'],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        conn.commit()
        return True, "Patient record saved successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def get_all_patients():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
        
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM patients ORDER BY created_at DESC", conn)
    conn.close()
    return df

def get_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'patient_id': row[1],
            'age': row[2],
            'gender': row[3],
            'sleep_duration': row[4],
            'sleep_efficiency': row[5],
            'rem_sleep': row[6],
            'deep_sleep': row[7],
            'light_sleep': row[8],
        }
    return None
