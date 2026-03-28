import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import time
import os

print("🚀 Booting up Continuous ML Training Microservice...")
os.makedirs('models', exist_ok=True)
model_path = 'models/rf_health_model.pkl'
telemetry_file = 'telemetry_log.csv'

while True:
    try:
        if os.path.exists(telemetry_file):
            df = pd.read_csv(telemetry_file)
            
            # Only train if we have a decent amount of data
            if len(df) > 10: 
                X = df[['cpu_load', 'temperature']]
                y = df['status']
                
                model = RandomForestClassifier(n_estimators=15, random_state=42)
                model.fit(X.values, y.values)
                
                joblib.dump(model, model_path)
                print(f"[{time.strftime('%H:%M:%S')}] 🧠 Model Retrained & Saved! (Total data points: {len(df)})")
        
        time.sleep(5)
        
    except Exception as e:
        print(f"Waiting for clean data... Error: {e}")
        time.sleep(5)